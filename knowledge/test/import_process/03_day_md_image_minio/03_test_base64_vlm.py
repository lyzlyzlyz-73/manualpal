import os

from openai import OpenAI
import base64
from dotenv import load_dotenv
load_dotenv()
# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),      # 老师已经配置在系统环境变量中了。
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 图片转 Base64
with open(r"C:\Users\ASUS\Desktop\shark.png", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

# 调用 VLM
response = client.chat.completions.create(
    model="qwen3.7-flash-2026-07-15",  # 视觉模型
    #model="qwen3-vl-plus", #视觉模型
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",           # 告诉 API：这是一段文字
                    "text": "请描述这张图片的内容,并仔细识别屏幕中央的物体或者生物，描述这是哪种地形，图片中可能是哪"
                },
                {
                    "type": "image_url",       # 告诉 API：这是一张图片
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=100
)

summary = response.choices[0].message.content
print(summary)
