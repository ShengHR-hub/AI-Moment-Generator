import requests

# 你的 FastAPI 服务地址
url = "http://127.0.0.1:8000/api/generate"

# 找一张你电脑上的测试图片 (替换为真实路径)
# 注意前面加一个小写的 r，这是为了防止 Windows 系统的斜杠 \ 产生转义错误
image_path = r"D:\测试图.jpg"

# 准备模拟前端传过去的文本和文件
data = {
    "user_direction": "周末出去玩，帮我写一段轻松幽默的文案"
}
files = {
    "image": ("test_image.jpg", open(image_path, "rb"), "image/jpeg")
}
print("正在发送请求到后端...")

try:
    # proxies={"http": None, "https": None} 可以强制绕过本地代理软件的干扰
    # timeout=10 表示如果 10 秒后后端还没反应，就直接报错，不再傻等
    response = requests.post(
        url, 
        data=data, 
        files=files, 
        proxies={"http": None, "https": None},
        timeout=10
    )
    print("状态码:", response.status_code)
    print("返回结果:", response.json())

except requests.exceptions.Timeout:
    print("❌ 请求超时了！请检查你的 FastAPI 服务端有没有运行，或者后端是否卡死了。")
except requests.exceptions.ConnectionError:
    print("❌ 连接被拒绝！你的 FastAPI 后端肯定没有启动。请先运行 uvicorn main:app --reload")
except Exception as e:
    print(f"❌ 发生了其他错误: {e}")
print("返回结果:", response.json())