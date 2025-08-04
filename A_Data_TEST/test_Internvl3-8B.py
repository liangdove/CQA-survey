import os
# 设置 Hugging Face 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from openai import OpenAI
import base64
import json
from PIL import Image
import io
import os
from tqdm import tqdm
import argparse
import time
from typing import List, Dict, Tuple

def image_to_base64(image_path):
    """Convert image file to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_qa_file(qa_file_path: str) -> List[Dict[str, str]]:
    """解析Q&A文件，返回包含图片名、问题和答案的列表"""
    qa_data = []
    
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_entry = {}
    for line in lines:
        line = line.strip()
        if line.startswith('I: '):
            if current_entry:  # 保存前一个条目
                qa_data.append(current_entry)
            current_entry = {'image': line[3:]}
        elif line.startswith('Q: '):
            current_entry['question'] = line[3:]
        elif line.startswith('A: '):
            current_entry['answer'] = line[3:]
    
    if current_entry:  # 保存最后一个条目
        qa_data.append(current_entry)
    
    return qa_data

def call_api_for_question(client, image_path: str, question: str, model_name: str):
    """调用API获取模型回答"""
    try:
        if not os.path.exists(image_path):
            return "图片文件不存在"
        
        base64_image = image_to_base64(image_path)
        
        # 准备提示词，参照test.py的格式
        text = f"{question}回答时仅用一个词或短语，如：“数字”、“职位名称”、“第三层/第二层/第一层”、“是/否”、“姓名”等简洁词汇回答问题。"

        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': text,
                }, {
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{base64_image}"
                    },
                }],
            }],
            temperature=0.1,
            top_p=0.9,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API调用错误: {str(e)}"

def compare_answers(model_answer: str, standard_answer: str) -> bool:
    """比较模型答案与标准答案，返回是否正确"""
    # 简单的字符串匹配，可以根据需要调整比较逻辑
    model_clean = model_answer.strip().lower()
    standard_clean = standard_answer.strip().lower()
    
    # 完全匹配
    if model_clean == standard_clean:
        return True
    
    # 如果标准答案包含在模型答案中，也认为正确
    if standard_clean in model_clean:
        return True
    
    return False

def evaluate_vqa_on_custom_dataset(
    qa_file_path: str, 
    images_dir: str, 
    output_file: str = 'results.json', 
    limit: int = None,
    api_key: str = 'YOUR_API_KEY',
    base_url: str = 'http://0.0.0.0:23333/v1'
):
    """在自定义VQA数据集上评估模型"""
    
    # 初始化OpenAI客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 获取模型名称
    try:
        model_name = client.models.list().data[0].id
        print(f"使用模型: {model_name}")
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        model_name = "default"
    
    # 解析Q&A文件
    qa_data = parse_qa_file(qa_file_path)
    print(f"共找到 {len(qa_data)} 个测试样本")
    
    if limit:
        qa_data = qa_data[:limit]
        print(f"限制测试样本数量为: {limit}")
    
    results = []
    correct_count = 0
    total_count = 0
    
    print(f"开始评估 {len(qa_data)} 个样本...")
    
    for i, entry in enumerate(tqdm(qa_data, desc="处理样本")):
        try:
            image_name = entry['image']
            question = entry['question']
            standard_answer = entry['answer']
            
            # 构建图片路径
            image_path = os.path.join(images_dir, image_name)
            
            # 调用API获取模型回答
            model_answer = call_api_for_question(client, image_path, question, model_name)
            
            # 比较答案
            is_correct = compare_answers(model_answer, standard_answer)
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # 记录结果 - 与test.py保持一致的格式
            result = {
                'image': image_name,
                'question': question,
                'standard_answer': standard_answer,
                'model_answer': model_answer,
                'is_correct': is_correct
            }
            results.append(result)
            
            # 打印进度信息
            if (i + 1) % 50 == 0:
                current_acc = correct_count / total_count
                print(f"已处理 {i+1}/{len(qa_data)}, 当前准确率: {current_acc:.4f}")
            
            # 避免API调用过快
            # time.sleep(0.2)
                
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            # 记录错误结果
            result = {
                'image': entry.get('image', ''),
                'question': entry.get('question', ''),
                'standard_answer': entry.get('answer', ''),
                'model_answer': f"处理错误: {str(e)}",
                'is_correct': False
            }
            results.append(result)
            total_count += 1
            continue
    
    # 计算最终指标
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # 保存结果
    evaluation_results = {
        'accuracy': accuracy,
        'total_samples': total_count,
        'correct_samples': correct_count,
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估完成！")
    print(f"总样本数: {total_count}")
    print(f"正确样本数: {correct_count}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"结果已保存到: {output_file}")
    
    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用LMDeploy API在自定义VQA数据集上评估模型')
    parser.add_argument('--qa_file', type=str, 
                       default="C:\\E\\CQA-survey\\A_Data_TEST\\data\\Q_and_A.txt",
                       help='Q&A数据集文件路径')
    parser.add_argument('--images_dir', type=str, 
                       default="C:\\E\\CQA-survey\\A_Data_TEST\\data\\org_chart_images",
                       help='图片文件夹路径')
    parser.add_argument('--output', type=str, 
                       default='lmdeploy_vqa_results.json', 
                       help='输出结果文件')
    parser.add_argument('--limit', type=int, default=None, 
                       help='限制测试样本数量')
    parser.add_argument('--api_key', type=str, default='YOUR_API_KEY',
                       help='API密钥')
    parser.add_argument('--base_url', type=str, default='http://0.0.0.0:23333/v1',
                       help='API服务地址')
    
    args = parser.parse_args()
    
    # 运行评估
    evaluate_vqa_on_custom_dataset(
        qa_file_path=args.qa_file,
        images_dir=args.images_dir,
        output_file=args.output,
        limit=args.limit,
        api_key=args.api_key,
        base_url=args.base_url
    )

# 使用示例:
# python chartqa_test_lmdeploy_api.py --qa_file "C:\E\CQA-survey\A_Data_TEST\data\Q_and_A.txt" --images_dir "C:\E\CQA-survey\A_Data_TEST\data\org_chart_images" --output "lmdeploy_results.json"

# 测试少量样本:
# python chartqa_test_lmdeploy_api.py --limit 10 --output "lmdeploy_test_results.json"

# 指定API服务地址:
# python chartqa_test_lmdeploy_api.py --base_url "http://localhost:23333/v1" --api_key "your_api_key"


