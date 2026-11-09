# LifeLinks 后端

App 现在要求登录（Sign in with Apple）才能进入，登录换取的 session token 同时也是访问
本后端「AI 整理」接口的鉴权凭证。记录、人物、关系、承诺、金额、备份等本地数据功能不依赖
本后端，只是必须先登录才能进 App。

## 本地启动

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填写 SESSION_JWT_SECRET、APPLE_BUNDLE_ID 和 DEEPSEEK_API_KEY（去 https://platform.deepseek.com 申请）
uvicorn main:app --reload --port 8000
```

## 在 App 中配置

首次打开 App 的登录页会要求填写后端地址（如 `http://127.0.0.1:8000`，真机调试需改为电脑的
局域网 IP），然后点「使用 Apple 登录」。登录成功后 App 会自动保存后端签发的 session token，
之后无需再手动填写。这个后端地址与令牌同样会同步到「我的」→「AI 设置」页面。

## 接口

`POST /auth/apple`
- Body: `{"identity_token": "<Sign in with Apple 返回的 identityToken>"}`
- 返回：`{"session_token": "...", "user_id": "<Apple sub>", "expires_at": 1700000000}`

`POST /extract` / `POST /extract_card` / `POST /summarize_person`
- Header: `Authorization: Bearer <session_token>`（来自 /auth/apple）
- `/extract_card`：识别名片等图片，Body 为 `{"image_base64": "...", "image_mime": "image/jpeg", "occurred_at_hint": "..."}`，
  走 DeepSeek 的视觉模型（DEEPSEEK_VISION_MODEL），不经过备用服务。返回里多一个 `clean_text`：
  去掉乱码后的备忘文字，App 用它替换本地 OCR 的文字。
- `/extract` 传 `"is_ocr_text": true` 时表示文字来自图片 OCR，同样返回 `clean_text`（图片识别失败时的退路）。

`POST /account/delete`
- Header: `Authorization: Bearer <session_token>`
- Body: `{"authorization_code": "<App 重新做 Apple 验证拿到的 code>"}`
- 必须配置 `APPLE_TEAM_ID` / `APPLE_KEY_ID` / `APPLE_PRIVATE_KEY`：用 code 换 token 并撤销 Apple 授权，
  再让该用户此前签发的 session token 失效。未配置时返回 503，不会报告删除成功。可用 `/health` 的 `apple_revoke_configured` 确认。
- 吊销记录写在 `REVOKED_SESSIONS_PATH`（只存用户标识的 SHA-256 摘要），必须放在持久磁盘上，
  否则重启或重新部署后，已删除账号的旧 token 会恢复有效。多实例部署需要改用数据库或 Redis。
- 返回：`{"deleted": true, "apple_revoked": true}`

`GET /privacy`、`GET /terms`：隐私政策与用户协议 HTML 页面，内容在 `legal.py`；
运营者与联系邮箱由环境变量 `OPERATOR_NAME`、`CONTACT_EMAIL` 提供。

`GET /health` 用于探活。

## 安全

- `DEEPSEEK_API_KEY` 只存在于服务端环境变量，不会出现在任何响应或日志中。
- `SESSION_JWT_SECRET` 未设置时 `check_auth` 会拒绝所有请求（本地调试可设 `ALLOW_INSECURE_DEV=1` 临时放行），正式对外部署前
  必须设置为随机字符串，且 `APPLE_BUNDLE_ID` 必须和 App 的 bundle id 一致。
- `/auth/apple` 会校验 Apple identity token 的签名（Apple 官方 JWKS）、`iss`、`aud`、`exp`，
  校验通过后才签发本服务自己的 session token，不会把 Apple 的 token 原样透传或存储。
