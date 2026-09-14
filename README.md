# LifeLinks 后端（可选，仅用于「服务端 AI 整理」）

App 在没有配置此后端的情况下，其他所有功能（记录、人物、关系、承诺、金额、备份）都完整可用，
只是无法使用「AI 整理」一键提取，需要手动整理。

## 本地启动

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填写 APP_SHARED_TOKEN 和 DEEPSEEK_API_KEY（去 https://platform.deepseek.com 申请）
uvicorn main:app --reload --port 8000
```

## 在 App 中配置

「我的」→「AI 设置」：
- 开启「启用服务端 AI 整理」
- 后端地址填 `http://127.0.0.1:8000`（真机调试需改为电脑的局域网 IP）
- 访问令牌填与 `.env` 中 `APP_SHARED_TOKEN` 相同的值

## 接口

`POST /extract`
- Header: `Authorization: Bearer <APP_SHARED_TOKEN>`
- Body: `{"text": "...", "occurred_at_hint": "2026-09-13T12:00:00Z"}`
- 返回：结构化的 persons / relationships / commitments / money_mentions

`GET /health` 用于探活。

## 安全

- `DEEPSEEK_API_KEY` 只存在于服务端环境变量，不会出现在任何响应或日志中。
- 未设置 `APP_SHARED_TOKEN` 时不做鉴权，仅适合本地开发，正式对外部署前必须设置。
