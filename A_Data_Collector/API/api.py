from openai import OpenAI
import base64
import os
import json
from pathlib import Path
import glob

def encode_image(image_path):
    """将图片编码为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_files(folder_path):
    """获取文件夹中所有图片文件"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension), recursive=False))
        # image_files.extend(glob.glob(os.path.join(folder_path, extension.upper()), recursive=False)) # linux系统区分大小写
    return image_files

def process_single_image(client, image_path, sys_msg, text):
    """处理单张图片"""
    try:
        encoded_image = encode_image(image_path)
        
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
        
        return chat_rsp.choices[0].message.content
    except Exception as e:
        return f"Error processing image: {str(e)}"

def batch_process_images(folder_path, output_json_path):
    """批量处理图片"""
    client = OpenAI(
        api_key="??",
        base_url="https://chat.intern-ai.org.cn/api/v1/",
    )
    
    sys_msg = "你是一个组织架构图解析专家，需根据提供的架构图生成问答对。每个问题必须基于图中信息，答案必须是**单个单词、数字或短语**（如数字、部门名称等），禁止完整句子。"
    text = """
    分析这个层级关系图，提出**一个**问题，并自己做出回答。

    **问题类型**：
    问题需从以下类型中选择：
    
   - 计数统计：如统计数量（例："市场部下有几个子部门？"）
   - 属性查询：如获取部门/职位属性（例："财务总监属于哪个部门？" ）
   - 节点识别：如询问特定职位/部门（例："CEO的直接上级是谁？"）
   - 层级判断：如确定层级深度（例："研发总监处于第几层级？"）
   - 是非判断：xx部门属于xx部门吗？（例："市场部属于销售部吗？"）
   - 关系判断：如确定部门之间的关系（例："市场部和销售部是同级部门吗？"）
   - 从属判断：如确定从属关系（例："xxx部门归yyy管理吗？"）
   - 全图分析：最高层级是什么？（例："这个组织架构图的最高层级是什么？"）

   注意问题必须是客观题，避免生成主观题或开放式问题。**避免生成多单词回答**的问题，如"市场部下有哪些子部门？"

   **答案类型**：
   - 单个单词或数字：如"CEO"、"3"、"市场部"等等。
   
   **输出格式**：
    请严格按照以下格式回答：
    Q: xxx?
    A: yyy
    """
    
    # 获取所有图片文件
    image_files = get_image_files(folder_path)
    total_images = len(image_files)
    
    if total_images == 0:
        print("未找到图片文件！")
        return
    
    print(f"找到 {total_images} 张图片，开始批量处理...")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n处理进度: {i}/{total_images} - {os.path.basename(image_path)}")
        
        response = process_single_image(client, image_path, sys_msg, text)
        
        result = {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "question": text,
            "response": response
        }
        
        results.append(result)
        
        print(f"模型回答: \n{response}")
    
    # 保存结果到JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n批量处理完成！结果已保存到: {output_json_path}")

if __name__ == "__main__":
    # 配置路径
    folder_path = "C:\\E\\CQA-Dataset\\A_Data_Collector\\org_chart_images_demo"
    output_json_path = "C:\\E\\CQA-Dataset\\A_Data_Collector\\batch_results.json"
    
    # 执行批量处理
    batch_process_images(folder_path, output_json_path)