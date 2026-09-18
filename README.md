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

`POST /extract` / `POST /summarize_person`
- Header: `Authorization: Bearer <session_token>`（来自 /auth/apple）

`GET /health` 用于探活。

## 安全

- `DEEPSEEK_API_KEY` 只存在于服务端环境变量，不会出现在任何响应或日志中。
- `SESSION_JWT_SECRET` 未设置时 `check_auth` 会直接放行（仅适合本地开发），正式对外部署前
  必须设置为随机字符串，且 `APPLE_BUNDLE_ID` 必须和 App 的 bundle id 一致。
- `/auth/apple` 会校验 Apple identity token 的签名（Apple 官方 JWKS）、`iss`、`aud`、`exp`，
  校验通过后才签发本服务自己的 session token，不会把 Apple 的 token 原样透传或存储。
