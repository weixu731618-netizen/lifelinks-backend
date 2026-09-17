"""
LifeLinks 后端 —— 面向 iOS App 的 AI 结构化提取服务。

只做一件事：接收一段中文原文，调用 DeepSeek 大模型，返回结构化的
人物（含姓名/性别/年龄或出生年月/职业/所属公司）/关系/承诺/金额建议。
真正写入 App 本地数据库前，所有建议都必须经过用户在 App 内确认。

安全说明：
- DeepSeek 的 API Key 只通过环境变量注入本进程，绝不返回给客户端，也不应写入任何日志。
- 客户端调用本服务需要携带 Authorization: Bearer <APP_SHARED_TOKEN>
  （与 DeepSeek 的 API Key 无关，只是「App-后端」之间的简单鉴权）。
"""
import os
import json
import time
import logging
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lifelinks")

APP_SHARED_TOKEN = os.environ.get("APP_SHARED_TOKEN", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
# deepseek-flash 默认开启思考模式（会先输出一大段 reasoning_content 再给正文，
# 计费和耗时都按这部分算）。这两个任务都是"照要求转换/生成"，用不上多步思考，
# 显式关掉换取更快更省的响应。见 https://api-docs.deepseek.com/guides/thinking_mode/
NON_THINKING = {"thinking": {"type": "disabled"}}
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "30"))

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
    hometown: Optional[str] = None        # 籍贯/常住地
    facts: list[str] = []                 # 其他无法归入固定字段的开放性事实，每条一句话


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


class SummarizePersonResponse(BaseModel):
    summary: str


def check_auth(authorization: Optional[str]):
    if not APP_SHARED_TOKEN:
        # 未配置令牌时，仅建议在本地开发环境使用，生产环境务必设置 APP_SHARED_TOKEN。
        return
    expected = f"Bearer {APP_SHARED_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="未授权：Authorization 令牌不正确")


@app.get("/health")
def health():
    return {"status": "ok", "ai_configured": bool(DEEPSEEK_API_KEY), "model": DEEPSEEK_MODEL}


@app.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest, authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)

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
        logger.info("extract 请求耗时 %.2fs", elapsed)

    return result


SUMMARIZE_SYSTEM_PROMPT = """你是一个帮用户总结"这个人"的助手，服务对象是靠人脉办事的人（销售/创业者/做社群等），
不是普通聊天助手。输入是关于某个人的结构化要点（基础信息、能提供什么、已知事实、承诺与人情往来情况），
输出一段 120-200 字的自然语言中文总结，直接给出结论性的判断供参考，不要用"可能""也许"这类模糊词回避判断。

要求：
1. 只根据给出的要点做总结/推断，不要编造要点之外的具体事实（姓名、公司、金额等）。
2. 总结要覆盖：这个人是谁、他能提供什么帮助、当前关系处于什么状态（比如是否有未还人情、
   未完成的承诺、多久没联系了）。哪一项要点里没有信息，就不提，不要为了凑内容编造。
3. 如果要点里明显有"欠人情""承诺逾期""很久没联系"这类信号，要直接指出，不要含糊其辞。
4. 只输出总结正文，不要输出标题、不要用 markdown、不要输出多余说明或免责声明。
"""


SYSTEM_PROMPT = """你是一个从中文日记式文本中抽取人物信息、人物关系、承诺和金额信息的助手。
只输出一个 JSON 对象，不要输出任何多余文字、不要用 markdown 代码块包裹。

JSON 结构：
{
  "persons": [{"name": "", "aliases": [], "how_met": "", "gender": "", "occupation": "",
               "organization": "", "birth_date": "YYYY-MM-DD", "relation_to_me": "", "hometown": "",
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
6. due_date_text 保留原文的模糊时间表达（如"下周五"、"月底"），不要自己换算成具体日期。
7. persons 列表不要包含"我"/说话人自己，只列出说话人之外提到的其他人物；但 relationships、
   commitments 中涉及"我"的地方，from_name/counterpart_name 可以正常写"我"。
8. kind 为 introduced_by 时，from_name 必须是"做介绍这个动作的人"，to_name 是"被介绍认识的人"。
   例如"老李介绍我认识老张"应输出 from_name="老李"、to_name="老张"（说明老李是介绍人，
   老张是被引荐对象），而不是以"我"作为 from_name，因为这样会丢失谁是介绍人这个关键信息。
9. relation_to_me 只在原文明确体现说话人与该人物的关系时填写（如"我同学老张""朋友老李"），
   没有明确体现就留空，不要瞎猜。
10. facts 用来装其他所有不属于 gender/occupation/organization/birth_date/hometown 的、
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
12. 判断 facts 该拆成几条时，先分清楚"关于这个人的不同维度信息"和"同一件事按先后顺序
    发生的经过"：前者才按第10条拆成多条，后者只算一件事，只写一条（或者根本不写进
    facts，因为原始记录里已经保留了完整经过）。例如"今天过来，约好4:56，6点还没来，
    后来回去了"是同一次赴约从开始到结束的时间线，不是关于这个人的4个不同事实，只应该
    写一条，例如"4:56有约但6点仍未到，后来离开了"，不要按"过来""约好4:56""6点没来"
    "回去了"这样按时间点逐句拆成多条。
"""


@app.post("/summarize_person", response_model=SummarizePersonResponse)
def summarize_person(payload: SummarizePersonRequest, authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)

    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="服务端未配置 DEEPSEEK_API_KEY，无法调用真实模型。请在 backend/.env 中配置后重启服务。",
        )

    start = time.monotonic()
    try:
        summary = call_deepseek_summary(payload.profile_text)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("AI 总结失败: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="AI 服务调用失败，请稍后重试。") from exc
    finally:
        elapsed = time.monotonic() - start
        logger.info("summarize_person 请求耗时 %.2fs", elapsed)

    return SummarizePersonResponse(summary=summary)


def call_deepseek_summary(profile_text: str) -> str:
    from openai import OpenAI  # 依赖见 requirements.txt

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=REQUEST_TIMEOUT_SECONDS)

    completion = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
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
