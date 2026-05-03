import os

# 腾讯云 COS 配置
COS_SECRET_ID = os.getenv("COS_SECRET_ID")
COS_SECRET_KEY = os.getenv("COS_SECRET_KEY")
COS_REGION = os.getenv("COS_REGION", "ap-beijing")
COS_BUCKET = os.getenv("COS_BUCKET")
COS_UPLOAD_PREFIX = "uploads/"

# 阿里云通义千问配置
QWEN_API_KEY = os.getenv("QWEN_API_KEY")