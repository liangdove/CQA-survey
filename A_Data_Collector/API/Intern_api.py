from openai import OpenAI
import base64
import os
import json
import time
import random
from pathlib import Path
import glob

def encode_image(image_path):
    """将图片编码为base64格式，增加错误处理"""
    try:
        if not os.path.exists(image_path):
            print(f"错误: 图片文件不存在 - {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"编码图片失败 {image_path}: {str(e)}")
        return None

def get_image_files(folder_path):
    """获取文件夹中所有图片文件，增加验证"""
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        return []
        
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension), recursive=False))
        # 处理大写扩展名
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper()), recursive=False))
    
    # 去重
    image_files = list(set(image_files))
    
    # 按照文件名中的数字进行排序
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        # 提取文件名中的数字部分
        import re
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    image_files.sort(key=extract_number)
    print(f"找到 {len(image_files)} 个图片文件，按数字顺序排序")
    return image_files

def process_single_image(client, image_path, sys_msg, text, max_retries=3):
    """处理单张图片，增加重试机制"""
    for attempt in range(max_retries):
        try:
            encoded_image = encode_image(image_path)
            if not encoded_image:
                return {"success": False, "error": "图片编码失败", "response": None}
            
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
            
            response = chat_rsp.choices[0].message.content
            
            # 每次成功请求后随机休息2-3秒
            delay = random.uniform(2.0, 3.0)
            print(f"API请求成功，休息 {delay:.1f} 秒...")
            time.sleep(delay)
            
            return {"success": True, "response": response, "attempt": attempt + 1}
            
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试失败: {str(e)}")
            if attempt < max_retries - 1:
                # 失败重试时也添加延迟，但稍长一些
                retry_delay = random.uniform(3.0, 5.0)
                print(f"等待 {retry_delay:.1f} 秒后重试...")
                time.sleep(retry_delay)
            else:
                return {"success": False, "error": str(e), "response": None}

def load_progress(progress_file):
    """加载处理进度"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            print(f"加载进度: 已处理 {len(progress.get('processed_files', []))} 张图片")
            return progress
        except Exception as e:
            print(f"加载进度文件失败: {str(e)}")
    return {"processed_files": [], "results": []}

def save_progress(progress_file, progress):
    """保存处理进度"""
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存进度失败: {str(e)}")

def validate_qa_response(response):
    """简单验证问答格式"""
    if not response:
        return False
    
    lines = response.strip().split('\n')
    has_question = any(line.strip().startswith('Q:') for line in lines)
    has_answer = any(line.strip().startswith('A:') for line in lines)
    
    return has_question and has_answer

def batch_process_images(folder_path, output_json_path, resume=True):
    """批量处理图片，增加进度保存和恢复功能"""
    # 验证输入路径
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # 进度文件路径
    progress_file = output_json_path.replace('.json', '_progress.json')
    
    # 加载进度
    progress = load_progress(progress_file) if resume else {"processed_files": [], "results": []}
    
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
   - 属性查询：如获取部门/职位属性（例："财务总监属于哪个部门？"  )
   - 层级判断：如确定层级深度（例："研发总监处于第几层级？"）
   - 是非判断：xx部门属于xx部门吗？（例："市场部属于销售部吗？"）
   - 关系判断：如确定部门之间的关系（例："市场部和销售部是同级部门吗？"）
   - 节点识别：如询问特定职位/部门（例："CEO的直接上级/部门是什么？"）
   - 从属判断：如确定从属关系（例："xxx部门归yyy管理吗？"）

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
    if not image_files:
        print("未找到图片文件！")
        return
    
    # 过滤已处理的文件
    if resume:
        processed_files = set(progress.get("processed_files", []))
        remaining_files = [f for f in image_files if f not in processed_files]
        print(f"总共 {len(image_files)} 张图片，剩余 {len(remaining_files)} 张待处理")
        image_files = remaining_files
    
    total_images = len(image_files)
    successful_count = 0
    failed_count = 0
    
    print(f"开始批量处理 {total_images} 张图片...")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n处理进度: {i}/{total_images} - {os.path.basename(image_path)}")
        
        # 处理单张图片
        result = process_single_image(client, image_path, sys_msg, text)
        
        # 构建结果数据
        result_data = {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "question": text,
            "response": result.get("response"),
            "success": result.get("success", False),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if result.get("success"):
            # 验证问答格式
            is_valid = validate_qa_response(result["response"])
            result_data["qa_format_valid"] = is_valid
            
            successful_count += 1
            print(f"成功处理 (尝试 {result.get('attempt', 1)} 次): \n{result['response']}")
            
            if not is_valid:
                print("警告: 响应格式可能不正确")
        else:
            result_data["error"] = result.get("error")
            failed_count += 1
            print(f"处理失败: {result.get('error')}")
        
        # 更新进度
        progress["results"].append(result_data)
        progress["processed_files"].append(image_path)
        
        # 每处理5张图片保存一次进度
        if i % 5 == 0 or i == total_images:
            save_progress(progress_file, progress)
            print(f"进度已保存: {i}/{total_images}")
    
    # 保存最终结果
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(progress["results"], f, ensure_ascii=False, indent=2)
        
        print(f"\n批量处理完成！结果已保存到: {output_json_path}")
        print(f"处理统计: 成功 {successful_count} 张，失败 {failed_count} 张")
        
        # 清理进度文件
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("进度文件已清理")
            
    except Exception as e:
        print(f"保存最终结果失败: {str(e)}")

if __name__ == "__main__":
    # 配置路径
    folder_path = "C:\\E\\CQA-Dataset\\A_Data_Collector\\org_chart_images"
    output_json_path = "C:\\E\\CQA-Dataset\\A_Data_Collector\\batch_results.json"
    
    # 执行批量处理 (resume=True 表示可以从中断处继续)
    batch_process_images(folder_path, output_json_path, resume=True)