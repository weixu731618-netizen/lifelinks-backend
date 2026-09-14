FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# 密钥（DEEPSEEK_API_KEY / APP_SHARED_TOKEN）通过部署平台的 secrets 注入，
# 不打进镜像、不写进这个文件、.env 不会被拷贝进来。
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
