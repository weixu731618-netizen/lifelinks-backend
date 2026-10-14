"""
LifeLinks 后端 —— 面向 iOS App 的 AI 结构化提取服务。

只做一件事：接收一段中文原文，调用 DeepSeek 大模型，返回结构化的
人物（含姓名/性别/年龄或出生年月/职业/所属公司）/关系/承诺/金额建议。
真正写入 App 本地数据库前，所有建议都必须经过用户在 App 内确认。

安全说明：
- DeepSeek 的 API Key 只通过环境变量注入本进程，绝不返回给客户端，也不应写入任何日志。
- 客户端先用 Sign in with Apple 拿到的 identity token 调 POST /auth/apple 换取本服务
  自己签发的 session token，之后所有请求都携带 Authorization: Bearer <session_token>。
  session token 与苹果的 identity token、DeepSeek 的 API Key 都无关，只是「App-后端」
  之间用来区分用户身份的简单鉴权，payload 里带的是苹果分配的稳定用户标识（sub）。
"""
import os
import json
import time
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import jwt
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

import legal

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lifelinks")

# 签发/校验「App-后端」session token 用的密钥，务必在生产环境设置为一个随机长字符串。
SESSION_JWT_SECRET = os.environ.get("SESSION_JWT_SECRET", "")
SESSION_TOKEN_TTL_SECONDS = int(os.environ.get("SESSION_TOKEN_TTL_SECONDS", str(90 * 24 * 3600)))
# 必须和 Xcode 里 PRODUCT_BUNDLE_IDENTIFIER 一致，否则会拒绝所有 Apple identity token。
APPLE_BUNDLE_ID = os.environ.get("APPLE_BUNDLE_ID", "com.weixu.lifelinks2.app")
APPLE_ISSUER = "https://appleid.apple.com"
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_REVOKE_URL = "https://appleid.apple.com/auth/revoke"
# 删除账号时撤销 Sign in with Apple 授权所需（Apple 开发者后台 → Keys 创建的 .p8 私钥）。
# 三项都配置了才会真正调用 Apple 的 revoke 接口；未配置时只做 session 校验并返回 apple_revoked=false。
APPLE_TEAM_ID = os.environ.get("APPLE_TEAM_ID", "9PG4KYBK7V")
APPLE_KEY_ID = os.environ.get("APPLE_KEY_ID", "")
APPLE_PRIVATE_KEY = os.environ.get("APPLE_PRIVATE_KEY", "").replace("\\n", "\n")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
# deepseek-flash 默认开启思考模式（会先输出一大段 reasoning_content 再给正文，
# 计费和耗时都按这部分算）。这两个任务都是"照要求转换/生成"，用不上多步思考，
# 显式关掉换取更快更省的响应。见 https://api-docs.deepseek.com/guides/thinking_mode/
NON_THINKING = {"thinking": {"type": "disabled"}}
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))

# 每用户调用频率限制（按 session token 里的用户标识计数）。
# 每分钟限制防止脚本或循环请求；每日限制（按北京时间自然日）控制总成本。
# 计数只保存在进程内存中，服务重启会清零；多实例部署需要改用数据库或 Redis。
RATE_LIMITS = {
    "extract": {"per_minute": 10, "per_day": 100},
    "summarize_person": {"per_minute": 5, "per_day": 30},
}
_rate_lock = threading.Lock()
_minute_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_day_hits: dict[tuple[str, str, str], int] = defaultdict(int)


def enforce_rate_limit(endpoint: str, user_id: str) -> None:
    """超过限制时抛出 429；未超过则记一次调用。应在调用 DeepSeek 之前执行。"""
    limits = RATE_LIMITS[endpoint]
    now = time.time()
    today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    with _rate_lock:
        window = _minute_hits[(endpoint, user_id)]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= limits["per_minute"]:
            raise HTTPException(status_code=429, detail="请求太频繁，请稍后再试。")

        day_key = (endpoint, user_id, today)
        if _day_hits[day_key] >= limits["per_day"]:
            raise HTTPException(status_code=429, detail="今天的 AI 调用次数已用完，明天再试。")

        window.append(now)
        _day_hits[day_key] += 1
        # 删除非今天的计数，避免内存只增不减
        for key in [k for k in _day_hits if k[2] != today]:
            del _day_hits[key]


app = FastAPI(title="LifeLinks Extraction Service")


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    occurred_at_hint: Optional[str] = None


class PersonOut(BaseModel):
    name: str
    aliases: list[str] = []
    how_met: Optional[str] = None
    gender: Optional[str] = None          # "男"/"女"/其他自由文本，无法判断则留空
    occupation: Optional[str] = None      # 职业/工作内容
    organization: Optional[str] = None    # 所属公司/组织
    birth_date: Optional[str] = None      # "YYYY-MM-DD"；只知道年月时日填 01；完全无法判断则留空，不得编造
    relation_to_me: Optional[str] = None  # 与说话人的关系标签，如"同学""朋友""同事""家人"
    # 与说话人的主要关系，固定取值，见 RELATION_LENSES 的 key；不确定或不在取值内一律为 None。
    primary_relation: Optional[str] = None
    hometown: Optional[str] = None        # 籍贯（老家），不是现居地
    residence: Optional[str] = None       # 常住地（现居城市）；搬家后以最新一次提到的为准
    facts: list[str] = []                 # 其他无法归入固定字段的开放性事实，每条一句话

    @field_validator("primary_relation", mode="before")
    @classmethod
    def _normalize_primary_relation(cls, value):
        # 模型偶尔会返回空串、大小写不同或取值之外的词，统一收敛成 None，不让脏值进到 App。
        if not isinstance(value, str):
            return None
        value = value.strip().lower()
        return value if value in RELATION_LENSES else None


class RelationshipOut(BaseModel):
    kind: str  # introduced_by/family/friend/colleague/partner/org_role/client/supplier/other
    from_name: str
    to_name: Optional[str] = None
    to_org_name: Optional[str] = None
    role_title: Optional[str] = None
    note: Optional[str] = None


class CommitmentOut(BaseModel):
    content: str
    direction: str  # mine/theirs/general
    counterpart_name: Optional[str] = None
    due_date_text: Optional[str] = None


class MoneyOut(BaseModel):
    amount_text: str
    currency: Optional[str] = None
    direction: str  # income/expense
    status: str  # occurred/expected/unconfirmed
    note: Optional[str] = None


class ExtractResponse(BaseModel):
    persons: list[PersonOut] = []
    relationships: list[RelationshipOut] = []
    commitments: list[CommitmentOut] = []
    money_mentions: list[MoneyOut] = []
    project_name_hint: Optional[str] = None


class SummarizePersonRequest(BaseModel):
    profile_text: str = Field(..., min_length=1, max_length=6000)
    # 用户与此人的主要关系，取值见 RELATION_LENSES；缺省或不认识的值一律按通用语气处理。
    primary_relation: Optional[str] = None


class SummarizePersonResponse(BaseModel):
    summary: str


class AppleAuthRequest(BaseModel):
    identity_token: str = Field(..., min_length=1)


class DeleteAccountRequest(BaseModel):
    # 删除前 App 会让用户重新做一次 Apple 登录，拿到新的 authorization code 用来撤销授权。
    authorization_code: Optional[str] = None


class DeleteAccountResponse(BaseModel):
    deleted: bool
    apple_revoked: bool


class AppleAuthResponse(BaseModel):
    session_token: str
    user_id: str
    expires_at: int


# 进程内缓存 Apple 的 JWKS（签名公钥集合），避免每次登录都请求一次。
_apple_jwks_cache: dict = {"keys": [], "fetched_at": 0.0}
_APPLE_JWKS_CACHE_TTL_SECONDS = 3600


def _get_apple_jwks() -> list[dict]:
    now = time.monotonic()
    if now - _apple_jwks_cache["fetched_at"] > _APPLE_JWKS_CACHE_TTL_SECONDS or not _apple_jwks_cache["keys"]:
        try:
            resp = httpx.get(APPLE_KEYS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            _apple_jwks_cache["keys"] = resp.json().get("keys", [])
            _apple_jwks_cache["fetched_at"] = now
        except Exception as exc:  # noqa: BLE001
            if _apple_jwks_cache["keys"]:
                logger.warning("刷新 Apple JWKS 失败，继续使用缓存: %s", type(exc).__name__)
            else:
                raise HTTPException(status_code=502, detail="无法获取 Apple 登录所需的公钥，请稍后重试。") from exc
    return _apple_jwks_cache["keys"]


def verify_apple_identity_token(identity_token: str) -> str:
    """校验 Sign in with Apple 返回的 identity token，返回苹果分配的稳定用户标识（sub）。"""
    try:
        header = jwt.get_unverified_header(identity_token)
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="无效的 Apple identity token") from exc

    kid = header.get("kid")
    matching_key = next((k for k in _get_apple_jwks() if k.get("kid") == kid), None)
    if matching_key is None:
        raise HTTPException(status_code=401, detail="无法匹配 Apple 签名公钥，请重新登录")

    try:
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(matching_key))
        claims = jwt.decode(
            identity_token,
            key=public_key,
            algorithms=["RS256"],
            audience=APPLE_BUNDLE_ID,
            issuer=APPLE_ISSUER,
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Apple identity token 校验失败：{exc}") from exc

    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Apple identity token 缺少用户标识")
    return sub


@app.post("/auth/apple", response_model=AppleAuthResponse)
def auth_apple(payload: AppleAuthRequest):
    if not SESSION_JWT_SECRET:
        raise HTTPException(status_code=503, detail="服务端未配置 SESSION_JWT_SECRET，无法签发登录凭证。")

    apple_sub = verify_apple_identity_token(payload.identity_token)

    now = int(time.time())
    expires_at = now + SESSION_TOKEN_TTL_SECONDS
    session_token = jwt.encode(
        {"sub": apple_sub, "iat": now, "exp": expires_at},
        SESSION_JWT_SECRET,
        algorithm="HS256",
    )
    logger.info("Apple 登录成功，用户 sub=%s", apple_sub)
    return AppleAuthResponse(session_token=session_token, user_id=apple_sub, expires_at=expires_at)


def check_auth(authorization: Optional[str]) -> str:
    """校验 App 自己签发的 session token，返回其中的用户标识（苹果 sub），用于按用户区分/记日志。"""
    if not SESSION_JWT_SECRET:
        # 默认拒绝：忘记配置密钥时不能让所有人免登录调用。本地调试需显式设置 ALLOW_INSECURE_DEV=1。
        if os.environ.get("ALLOW_INSECURE_DEV") == "1":
            return "dev"
        raise HTTPException(status_code=503, detail="服务端未配置 SESSION_JWT_SECRET，已拒绝请求。")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未授权：缺少登录凭证，请先完成 Apple 登录")
    token = authorization.removeprefix("Bearer ")
    try:
        claims = jwt.decode(token, SESSION_JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="登录凭证无效或已过期，请重新登录") from exc
    return claims.get("sub", "unknown")


def _apple_revoke_configured() -> bool:
    return bool(APPLE_TEAM_ID and APPLE_KEY_ID and APPLE_PRIVATE_KEY)


def _apple_client_secret() -> str:
    now = int(time.time())
    return jwt.encode(
        {"iss": APPLE_TEAM_ID, "iat": now, "exp": now + 600, "aud": APPLE_ISSUER, "sub": APPLE_BUNDLE_ID},
        APPLE_PRIVATE_KEY,
        algorithm="ES256",
        headers={"kid": APPLE_KEY_ID},
    )


def revoke_apple_authorization(authorization_code: str, expected_sub: str) -> None:
    """用 authorization code 换 token，确认是同一个用户后撤销本 App 在苹果处的授权。"""
    client_secret = _apple_client_secret()
    try:
        token_resp = httpx.post(
            APPLE_TOKEN_URL,
            data={
                "client_id": APPLE_BUNDLE_ID,
                "client_secret": client_secret,
                "code": authorization_code,
                "grant_type": "authorization_code",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="无法连接 Apple 服务，请稍后重试。") from exc
    if token_resp.status_code != 200:
        logger.warning("Apple token 交换失败: HTTP %s", token_resp.status_code)
        raise HTTPException(status_code=400, detail="Apple 授权码无效或已过期，请重试。")

    token_data = token_resp.json()
    id_token = token_data.get("id_token")
    if id_token:
        # 此 id_token 是 TLS 直连 Apple 的响应，这里只用来核对用户一致性，不需要再验签。
        sub = jwt.decode(id_token, options={"verify_signature": False}).get("sub")
        if sub != expected_sub:
            raise HTTPException(status_code=403, detail="Apple 账号与当前登录账号不一致。")

    token = token_data.get("refresh_token") or token_data.get("access_token")
    hint = "refresh_token" if token_data.get("refresh_token") else "access_token"
    if not token:
        raise HTTPException(status_code=502, detail="Apple 未返回可撤销的凭证，请稍后重试。")
    try:
        revoke_resp = httpx.post(
            APPLE_REVOKE_URL,
            data={
                "client_id": APPLE_BUNDLE_ID,
                "client_secret": client_secret,
                "token": token,
                "token_type_hint": hint,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="无法连接 Apple 服务，请稍后重试。") from exc
    if revoke_resp.status_code != 200:
        logger.warning("Apple revoke 失败: HTTP %s", revoke_resp.status_code)
        raise HTTPException(status_code=502, detail="撤销 Apple 授权失败，请稍后重试。")


@app.post("/account/delete", response_model=DeleteAccountResponse)
def delete_account(payload: DeleteAccountRequest, authorization: Optional[str] = Header(default=None)):
    """
    删除账号。服务端不保存任何用户内容数据，所以这里没有数据库记录要删：
    要做的是撤销 Apple 授权（App Store 审核 5.1.1(v) 要求），之后该 session 对应的用户需重新登录才会再次授权。
    """
    user_id = check_auth(authorization)

    revoked = False
    if _apple_revoke_configured():
        if not payload.authorization_code:
            raise HTTPException(status_code=400, detail="缺少 Apple 授权码，请重新验证身份后再删除账号。")
        revoke_apple_authorization(payload.authorization_code, user_id)
        revoked = True
    else:
        logger.warning("未配置 APPLE_TEAM_ID/APPLE_KEY_ID/APPLE_PRIVATE_KEY，跳过撤销 Apple 授权")

    logger.info("账号已删除，用户 sub=%s，apple_revoked=%s", user_id, revoked)
    return DeleteAccountResponse(deleted=True, apple_revoked=revoked)


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page():
    return legal.privacy_html()


@app.get("/terms", response_class=HTMLResponse)
def terms_page():
    return legal.terms_html()


@app.get("/health")
def health():
    return {"status": "ok", "ai_configured": bool(DEEPSEEK_API_KEY), "model": DEEPSEEK_MODEL}


@app.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest, authorization: Optional[str] = Header(default=None)):
    user_id = check_auth(authorization)
    enforce_rate_limit("extract", user_id)

    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="服务端未配置 DEEPSEEK_API_KEY，无法调用真实模型。请在 backend/.env 中配置后重启服务，"
                   "或在 App 中改用手动整理。",
        )

    start = time.monotonic()
    try:
        result = call_deepseek(payload.text, payload.occurred_at_hint)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("AI 提取失败: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="AI 服务调用失败，请稍后重试。") from exc
    finally:
        elapsed = time.monotonic() - start
        logger.info("extract 请求耗时 %.2fs，用户 sub=%s", elapsed, user_id)

    return result


SUMMARIZE_SYSTEM_PROMPT = """你在帮用户记住身边的人。用户会在见面或联系某个人之前看这段话，
目的是想起他是谁、他在意什么，让彼此的关系更近一步。输入是关于某个人的结构化要点
（基础信息、已知事实、承诺与往来情况），输出一段 100-160 字的中文总结。
语气像一位了解用户的朋友在轻声提醒：自然、具体、有温度，不要像报告或评估。

要求：
1. 只根据给出的要点，不要编造要点之外的任何内容，包括具体事实（姓名、公司、金额、经历）
   和心理、情绪（例如“你心里一直记着”“他应该很在意”“家里也惦记着”）。
   哪一项要点里没有信息，就不提，不要为了凑内容编造。
2. 内容顺序：先说他是谁、最近在意什么或经历了什么（要点里有才说，是否能提私人信息以“关系说明”为准）；
   再说你们之间的状态（上次联系是多久前、有没有没兑现的约定）。
   不要提“他能提供什么”“他有什么资源”。
3. 没兑现的承诺、很久没联系：只照实陈述事实（例如“原定九月底，还没有回音”），
   不评判，不吓唬用户，不使用“逾期”“转冷”“风险”“需尽快处理”这类词。
4. 总结到事实和状态为止，不要给行动建议，不要以“可以”“不妨”“建议”之类的话收尾。
5. 有信息的地方直接说，不要用“可能”“也许”绕开；没有信息的地方不要猜。
6. 只输出总结正文，不要输出标题、不要用 markdown、不要输出多余说明或免责声明。
"""

# 用户与此人的主要关系 -> 追加到总结 prompt 末尾的一小段语气说明。
# key 是与 App 约定的英文取值；缺省或不认识的取值走 NEUTRAL_LENS，由代码选择，不让模型自行判断关系。
NEUTRAL_LENS = (
    "用户与这个人的关系尚未确定：用中性语气，不要假设你们很亲近，也不要假设对方是客户或上级。"
    "要点里有他的家人、健康等情况，照常提，这是让人感到被记得的地方。"
)
RELATION_LENSES = {
    "friend": (
        "用户与这个人是朋友：语气随意自然，像朋友之间聊天。可以提他的家人、生活和近况。"
    ),
    "family": (
        "用户与这个人是亲属：像家里人说话，温和。可以提健康、孩子和长辈。"
        "不要使用“跟进”“推进”“对接”“答复”“回音”这类办事的词。"
    ),
    "superior": (
        "这个人是用户的上级（领导、老板，也包括老师、导师）：称呼对方时带上职务或身份，用“你”称呼用户。"
        "先谈他的工作、目标和压力（老师、导师则谈他的研究和对用户的指导），以及用户答应他的事。"
        "语气尊重、有分寸，不要过分亲昵。要点里有他的家人、健康等情况，也要简短提到，口吻是惦记，不是打探。"
    ),
    "client": (
        "这个人是用户的客户：先谈他的事顺不顺、他的问题有没有解决，以及用户答应他的事。"
        "语气真诚，不要显得在“经营”他。要点里有他的家人、健康、孩子等情况，也要提到，"
        "这正是让对方感到被记得的地方（例如他生病了、孩子要高考）。"
    ),
    "partner": (
        "这个人是用户的合作方（供应商、合伙人、投资人、渠道等）：先把约定、交付、付款或合作的进展说清楚，"
        "这是对方最在意的。语气平等、守信用。要点里有他的家人、健康等情况，在最后简短提一句。"
    ),
    "colleague": (
        "这个人是用户的同事：先谈共事的情况和对方在工作上的难处，公私之间留一点距离。"
        "要点里有他的家人、健康等情况（例如孩子刚出生、身体不舒服），也要简短提到，口吻是关心。"
    ),
    "subordinate": (
        "这个人是用户的下属（用户带的人）：先谈他的工作情况、成长和难处，以及用户答应他的事。"
        "语气关心，不居高临下，不说教。要点里有他的家人、健康等情况，也要提到，口吻是关心。"
    ),
}


def build_summary_system_prompt(primary_relation: Optional[str]) -> str:
    lens = RELATION_LENSES.get((primary_relation or "").strip(), NEUTRAL_LENS)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    return (
        f"{SUMMARIZE_SYSTEM_PROMPT}\n"
        f"今天的日期是 {today}。判断“多久没联系”“承诺是否已过期”时以此为准，"
        f"用“三个多月”“上周”这样的说法，不要只罗列日期。\n"
        f"关系说明：{lens}\n"
    )


SYSTEM_PROMPT = """你是一个从中文日记式文本中抽取人物信息、人物关系、承诺和金额信息的助手。
只输出一个 JSON 对象，不要输出任何多余文字、不要用 markdown 代码块包裹。

JSON 结构：
{
  "persons": [{"name": "", "aliases": [], "how_met": "", "gender": "", "occupation": "",
               "organization": "", "birth_date": "YYYY-MM-DD", "relation_to_me": "",
               "primary_relation": "friend|family|colleague|superior|subordinate|client|partner", "hometown": "",
               "residence": "",
               "facts": []}],
  "relationships": [{"kind": "introduced_by|family|friend|colleague|partner|org_role|client|supplier|other",
                      "from_name": "", "to_name": "", "to_org_name": "", "role_title": "", "note": ""}],
  "commitments": [{"content": "", "direction": "mine|theirs|general", "counterpart_name": "", "due_date_text": ""}],
  "money_mentions": [{"amount_text": "", "currency": "", "direction": "income|expense",
                       "status": "occurred|expected|unconfirmed", "note": ""}],
  "project_name_hint": ""
}

规则：
1. 只提取原文明确提到或能直接合理推断的信息，绝不编造姓名、年龄、公司等具体事实。
2. 不确定的字段留空字符串或省略，不要瞎猜一个具体值（尤其是 birth_date、amount_text）。
3. person 的 birth_date 只有在原文明确给出年份时才填写，只知道年龄时不要反推具体出生日期。
4. relationship 的 kind 必须是给定枚举之一，无法归类用 other。
5. 每个人只出现一次在 persons 中，即使多次提到。
   例外：同一段文字里如果明显是几个不同的人但称呼相同（如"王总""李哥"出现多个，
   原文用公司、职业、地点等区分了他们），必须拆成多个 person，name 在称呼后用全角括号加上
   最能区分的特征，例如"王总（万科）""王总（做建材的）"；relationships 的 from_name/to_name、
   commitments 的 counterpart_name 里引用他们时，必须写成与 persons 中完全一致的 name。
   原文没有任何可区分信息、无法判断是几个人时，当作同一个人，不要硬拆。
6. due_date_text 保留原文的模糊时间表达（如"下周五"、"月底"），不要自己换算成具体日期。
7. persons 列表不要包含"我"/说话人自己，只列出说话人之外提到的其他人物；但 relationships、
   commitments 中涉及"我"的地方，from_name/counterpart_name 可以正常写"我"。
8. kind 为 introduced_by 时，from_name 必须是"做介绍这个动作的人"，to_name 是"被介绍认识的人"。
   例如"老李介绍我认识老张"应输出 from_name="老李"、to_name="老张"（说明老李是介绍人，
   老张是被引荐对象），而不是以"我"作为 from_name，因为这样会丢失谁是介绍人这个关键信息。
9. relation_to_me 只在原文明确体现说话人与该人物的关系时填写（如"我同学老张""朋友老李"），
   没有明确体现就留空，不要瞎猜。
9a. primary_relation 是说话人与该人物最主要的关系，只能取下面七个值之一：
    friend（朋友、同学等平辈的私人关系）、family（亲属、伴侣）、colleague（同事）、
    superior（上级、领导、老板、老师、导师）、subordinate（下属、自己带的人）、
    client（付钱给说话人或其公司的客户，包括正在谈的潜在客户）、
    partner（合作方：供应商、合伙人、投资人、渠道等，和说话人对等合作或由说话人付钱的人）。
    只有原文明确体现了这种关系才填；没有明确体现，或同时体现了几种身份、无法判断哪个最主要时，
    留空字符串，不要猜。“某某介绍认识的”只说明怎么认识，不算关系。
    例如“我同学老张”填 friend；“我们公司的客户王总”填 client；“我老板”填 superior；
    “我带的实习生小王”填 subordinate；“我们的供应商老刘”填 partner。
    一个人同时有两种以上不同类别的身份时，必须留空，让用户自己决定，不要替用户选一个：
    例如“我大学同学老周现在是我们的客户”（同学属于 friend，客户属于 client）→ primary_relation 留空，
    同时把 relation_to_me 写成“大学同学，现在是客户”。同类身份不算冲突，如“同学加朋友”都是 friend。
9b. hometown 只填籍贯/老家（"浙江人""老家四川"）；residence 只填现在住在哪个城市
   （"住在杭州""常驻深圳""搬到上海了"）。两者不要混填：原文只说"在杭州"且看不出是老家
   还是现居地时，按现居地填 residence，hometown 留空。organization、occupation、residence
   原文表达了变动（"跳槽去了B公司""现在转做销售""搬到成都了"）时，填变动之后的最新值。
10. facts 用来装其他所有不属于 gender/occupation/organization/birth_date/hometown/residence 的、
    关于这个人的具体事实，只要原文提到就写一条，每条一句简短的话，不要归纳总结，
    也不要编造。参考但不限于这些维度：婚姻状况、子女情况、学历/毕业院校、行业、职级、
    副业、收入、兴趣爱好、运动习惯、饮食偏好、宠物、旅行偏好、性格特点、沟通偏好、
    健康状况、生日、纪念日，以及任何其他值得记住的具体信息（如"提到过想换工作"
    "孩子今年高考"）。原文没提到的维度完全不用出现在 facts 里，不要为了凑齐维度而编造。
11. 一件事如果同时出现"这件事本身发生的时间"和"希望被提醒的时间"这两个不同的时间点
    （比如"3:50放学，然后下午3:40提醒我过去"），只生成一条 commitment，绝不能拆成两条：
    - due_date_text 必须填"希望被提醒的那个时间点"（用户真正想被闹一下的时刻），
      不要填事情本身发生的时间。
    - content 里把事情本身的时间作为背景说清楚，例如"3:50接徐小朵放学（提前在3:40提醒）"。
    只有单一时间点、不存在"事件时间 vs 提醒时间"这种区分时，按原来的方式处理即可。
13. commitments 只记录"跟我有关、将来需要提醒我"的事：我要做的事（direction=mine）、
    对方答应我或欠我的事（direction=theirs，例如"他答应下周还我100美金"）、我和对方的约定或
    需要跟进的事（direction=general）。以下情况一律不要写进 commitments：
    - 对方自己做的、与我无关的事，例如"张大伟昨天空投领了100美金"，这只是关于张大伟的信息，
      写进这个人的 facts 即可；
    - 已经发生的事（时间在过去），例如"昨天……"，它是事实，不是需要提醒的事项。
    只有一个人自己的事（我要吃药、我要交材料），没有对方时，counterpart_name 留空。
14. money_mentions 只记录涉及"我"的收支（我收到、我付出、对方欠我或我欠对方）。别人自己的收入
    或花费（如"张大伟领了100美金空投"）与我无关，不要写进 money_mentions。
12. 判断 facts 该拆成几条时，先分清楚"关于这个人的不同维度信息"和"同一件事按先后顺序
    发生的经过"：前者才按第10条拆成多条，后者只算一件事，只写一条（或者根本不写进
    facts，因为原始记录里已经保留了完整经过）。例如"今天过来，约好4:56，6点还没来，
    后来回去了"是同一次赴约从开始到结束的时间线，不是关于这个人的4个不同事实，只应该
    写一条，例如"4:56有约但6点仍未到，后来离开了"，不要按"过来""约好4:56""6点没来"
    "回去了"这样按时间点逐句拆成多条。
"""


@app.post("/summarize_person", response_model=SummarizePersonResponse)
def summarize_person(payload: SummarizePersonRequest, authorization: Optional[str] = Header(default=None)):
    user_id = check_auth(authorization)
    enforce_rate_limit("summarize_person", user_id)

    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="服务端未配置 DEEPSEEK_API_KEY，无法调用真实模型。请在 backend/.env 中配置后重启服务。",
        )

    start = time.monotonic()
    try:
        summary = call_deepseek_summary(payload.profile_text, payload.primary_relation)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("AI 总结失败: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="AI 服务调用失败，请稍后重试。") from exc
    finally:
        elapsed = time.monotonic() - start
        logger.info("summarize_person 请求耗时 %.2fs，用户 sub=%s", elapsed, user_id)

    return SummarizePersonResponse(summary=summary)


def call_deepseek_summary(profile_text: str, primary_relation: Optional[str] = None) -> str:
    from openai import OpenAI  # 依赖见 requirements.txt

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS)

    completion = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": build_summary_system_prompt(primary_relation)},
            {"role": "user", "content": profile_text},
        ],
        temperature=0.3,
        max_tokens=600,
        extra_body=NON_THINKING,
    )
    content = completion.choices[0].message.content or ""
    if not content.strip():
        raise HTTPException(status_code=502, detail="AI 未能在限定长度内给出总结，请稍后重试。")
    return content.strip()


def call_deepseek(text: str, occurred_at_hint: Optional[str]) -> ExtractResponse:
    """
    调用 DeepSeek（OpenAI 兼容接口）做结构化信息抽取。
    """
    from openai import OpenAI  # 依赖见 requirements.txt

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS)

    user_prompt = f"发生时间参考：{occurred_at_hint or '未知'}\n原文：\n{text}"

    completion = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=8000,
        extra_body=NON_THINKING,
    )

    raw_text = completion.choices[0].message.content or "{}"
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="AI 返回结果不是合法 JSON，请重试。") from exc

    return ExtractResponse(**data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
