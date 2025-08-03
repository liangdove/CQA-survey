from openai import OpenAI
import base64

def encode_image(image_path):
    """将图片编码为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = OpenAI(
    api_key="??",  # 此处传token，不带Bearer
    base_url="https://chat.intern-ai.org.cn/api/v1/",
)

# 系统
sys_msg = "你是企业层级图分析专家，擅长分析组织架构图。请根据提供的图片内容进行分析和回答。"

# 用户
image_path = "C:\\E\\CQA-Dataset\\A_Data_Collector\\org_chart_images_from_urls\\1-200P40S141b2.png"  # 替换为实际图片路径
encoded_image = encode_image(image_path)
text = "项目工程部部长主管哪些部门？"

chat_rsp = client.chat.completions.create(
    model="internvl3-latest",
    messages=[
        {
            "role": "system",
            "content": sys_msg
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ],
    temperature=0.8,
    top_p=0.9,
    max_tokens=100
)

for choice in chat_rsp.choices:
    print(choice.message.content)