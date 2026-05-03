import uuid
import os
from contextlib import asynccontextmanager

import dashscope
from dashscope import MultiModalConversation
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qcloud_cos import CosConfig, CosS3Client

# 从你的 config.py 中导入新增的 QWEN_API_KEY
from config import (
    COS_SECRET_ID,
    COS_SECRET_KEY,
    COS_REGION,
    COS_BUCKET,
    COS_UPLOAD_PREFIX,
    QWEN_API_KEY,
)

# ---------------------------------------------------------------------------
# 初始化与配置
# ---------------------------------------------------------------------------
cos_client: CosS3Client | None = None
# 初始化通义千问 API 密钥
dashscope.api_key = QWEN_API_KEY

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化 COS 客户端。"""
    global cos_client
    config = CosConfig(
        Region=COS_REGION,
        SecretId=COS_SECRET_ID,
        SecretKey=COS_SECRET_KEY,
    )
    cos_client = CosS3Client(config)
    print("✅ 腾讯云 COS 客户端已初始化")
    yield
    cos_client = None
    print("🔒 腾讯云 COS 客户端已关闭")

app = FastAPI(title="AI Moment Caption Generator", version="0.1.1", lifespan=lifespan)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 核心逻辑函数
# ---------------------------------------------------------------------------
def upload_file_to_cos(file_bytes: bytes, filename: str) -> str:
    """将文件上传到 COS 并返回公网 URL"""
    ext = os.path.splitext(filename)[1]
    key = f"{COS_UPLOAD_PREFIX}{uuid.uuid4().hex}{ext}"
    cos_client.put_object(
        Bucket=COS_BUCKET,
        Body=file_bytes,
        Key=key,
    )
    return f"https://{COS_BUCKET}.cos.{COS_REGION}.myqcloud.com/{key}"

def call_qwen_ai(image_url: str, user_direction: str) -> str:
    """调用通义千问多模态模型生成文案"""
    messages = [
        {
            "role": "system",
            "content": [{"text": "你是一个精通中国年轻人互联网语境的社交媒体文案大师。请观察图片并结合用户要求，生成3条风格迥异、带有Emoji且自然的微信朋友圈文案。"}]
        },
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": f"用户对文案的具体要求或描述是：{user_direction}。请开始生成。"}
            ]
        }
    ]
    
    response = MultiModalConversation.call(
        model='qwen-vl-max',
        messages=messages
    )

    if response.status_code == 200:
        # 提取模型生成的文本内容
        return response.output.choices[0].message.content[0]['text']
    else:
        raise Exception(f"AI 生成失败: {response.code} - {response.message}")

# ---------------------------------------------------------------------------
# 业务接口
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate(
    image: UploadFile = File(..., description="用户上传的图片文件"),
    user_direction: str = Form(..., description="用户期望的文案风格或补充描述"),
):
    # 1. 读取并校验图片
    file_bytes = await image.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="上传的图片文件为空")

    try:
        # 2. 图片云端化：上传到 COS 获取公网访问 URL
        cos_url = upload_file_to_cos(file_bytes, image.filename)
        
        # 3. AI 创作：将图片 URL 和用户指令喂给多模态大模型
        ai_copywriting = call_qwen_ai(cos_url, user_direction)
        
        # 日志打印（便于开发调试）
        print("-" * 30)
        print(f"DEBUG: 图片已托管至 -> {cos_url}")
        print(f"DEBUG: 最终生成文案 -> \n{ai_copywriting}")
        print("-" * 30)

        # 4. 返回完整数据给前端
        return {
            "code": 0,
            "message": "success",
            "data": {
                "cos_url": cos_url,
                "user_direction": user_direction,
                "ai_copywriting": ai_copywriting
            },
        }

    except Exception as e:
        # 统一错误处理
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}