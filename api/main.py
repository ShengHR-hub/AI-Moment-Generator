import uuid
import os
import asyncio

import dashscope
from dashscope import MultiModalConversation
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qcloud_cos import CosConfig, CosS3Client

# 从你的 config.py 中导入 API 密钥
from api.config import (
    COS_SECRET_ID,
    COS_SECRET_KEY,
    COS_REGION,
    COS_BUCKET,
    COS_UPLOAD_PREFIX,
    QWEN_API_KEY,
)

# ---------------------------------------------------------------------------
# 初始化与配置 (移除 lifespan，改用懒加载)
# ---------------------------------------------------------------------------
cos_client: CosS3Client | None = None
dashscope.api_key = QWEN_API_KEY

def get_cos_client():
    """懒加载获取 COS 客户端，专治 Serverless 环境"""
    global cos_client
    if cos_client is None:
        config = CosConfig(
            Region=COS_REGION,
            SecretId=COS_SECRET_ID,
            SecretKey=COS_SECRET_KEY,
        )
        cos_client = CosS3Client(config)
        print("✅ 腾讯云 COS 客户端已初始化 (懒加载)")
    return cos_client

# 移除 lifespan 参数
app = FastAPI(title="AI Moment Caption Generator", version="0.2.0")

# CORS 配置保持不变
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
    
    # 每次调用时获取客户端
    client = get_cos_client()
    client.put_object(
        Bucket=COS_BUCKET,
        Body=file_bytes,
        Key=key,
    )
    return f"https://{COS_BUCKET}.cos.{COS_REGION}.myqcloud.com/{key}"


def call_qwen_ai(image_urls: list[str], user_direction: str) -> str:
    """调用通义千问多模态模型生成文案，支持多图"""
    # 构建 user content：每张图片一个 image 块 + 最后一个 text 块
    user_content = []
    for url in image_urls:
        user_content.append({"image": url})
    user_content.append({"text": f"用户对文案的具体要求或描述是：{user_direction}。请结合所有图片生成文案。"})

    messages = [
        {
            "role": "system",
            "content": [{"text": "你是一个精通中国年轻人互联网语境的社交媒体文案大师。请观察用户提供的所有图片并结合用户要求，生成3条风格迥异、带有Emoji且自然的微信朋友圈文案。请综合所有图片的内容来生成，让文案能够涵盖多张图片的共同主题或故事。"}]
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    response = MultiModalConversation.call(
        model='qwen-vl-max',
        messages=messages
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content[0]['text']
    else:
        raise Exception(f"AI 生成失败: {response.code} - {response.message}")


# ---------------------------------------------------------------------------
# 业务接口
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate(
    images: list[UploadFile] = File(..., description="用户上传的图片文件（支持多张）"),
    user_direction: str = Form(..., description="用户期望的文案风格或补充描述"),
):
    # 1. 校验图片数量
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="请至少上传一张图片")
    
    if len(images) > 9:
        raise HTTPException(status_code=400, detail="最多支持上传 9 张图片")

    try:
        # 2. 读取所有图片并上传到 COS
        cos_urls = []
        for image in images:
            file_bytes = await image.read()
            if not file_bytes:
                continue
            cos_url = upload_file_to_cos(file_bytes, image.filename)
            cos_urls.append(cos_url)

        if not cos_urls:
            raise HTTPException(status_code=400, detail="所有上传的图片文件均为空")

        # 3. AI 创作：将所有图片 URL 和用户指令喂给多模态大模型
        ai_copywriting = call_qwen_ai(cos_urls, user_direction)
        
        # 4. 返回完整数据给前端
        return {
            "code": 0,
            "message": "success",
            "data": {
                "cos_urls": cos_urls,
                "user_direction": user_direction,
                "ai_copywriting": ai_copywriting
            },
        }

    except Exception as e:
        # 统一错误处理
        print(f"❌ 运行异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}