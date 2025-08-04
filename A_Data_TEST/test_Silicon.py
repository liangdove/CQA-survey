import time
from openai import OpenAI
import base64
import os
import json
from typing import List, Dict, Tuple

MODEL = "Pro/THUDM/GLM-4.1V-9B-Thinking"

def encode_image(image_path):
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

def call_api_for_question(client, image_path: str, question: str):
    """调用API获取模型回答"""
    try:
        if not os.path.exists(image_path):
            return "图片文件不存在"
        
        base64_image = encode_image(image_path)
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "你是专业的组织架构图分析助手。请基于图像内容，准确识别部门的层级关系、职位信息或上下级关系。回答时请仅使用一个词或短语，不得添加解释或多余语句。回答示例包括：“3”、“董事长”、“第二层”、“是”、“张三”、“人力资源部”等。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question + "回答时仅用一个词或短语，回答类型如：“数字”、“职位名称”、“第三层/第二层/第一层”、“是/否”、“姓名”等简洁词汇回答问题。回答时需要根据图像内容给出具体的回答（3、张三、董事长、财务处等）。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API调用错误: {str(e)}"

def compare_answers(model_answer: str, standard_answer: str):
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

def test_model_on_dataset(qa_file_path: str, images_dir: str):
    """在整个数据集上测试模型并计算准确率"""
    
    # 硅基流动 客户端
    client = OpenAI(
        api_key="xxx", 
        base_url="https://api.siliconflow.cn/v1"
    )
    
    # InternVL 客户端
    # client = OpenAI(
    #     api_key="eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI0MzMwNjU4NyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1NDA1NDk1MiwiY2xpZW50SWQiOiJlYm1ydm9kNnlvMG5semFlazF5cCIsInBob25lIjoiMTUxMzA4MTA2MzIiLCJvcGVuSWQiOm51bGwsInV1aWQiOiJjZGVkZDI5My01ZjcxLTQ2YmItYjFkOS05NDhhMGRlNjgyMDYiLCJlbWFpbCI6ImppYWxlX2xpYW5nQGJ1cHQuZWR1LmNuIiwiZXhwIjoxNzY5NjA2OTUyfQ.2boot_RQMhfsBNhAUjwfdPxJTVDVe5bLcBDFmw0mZ12AVvL4WXiTwjwN-_LhDUjnmQbqmypBTyMCKXY_pWFs0A", 
    #     base_url="https://chat.intern-ai.org.cn/api/v1/"
    # )
    
    # 解析Q&A文件
    qa_data = parse_qa_file(qa_file_path)
    print(f"共找到 {len(qa_data)} 个测试样本")
    
    results = []
    correct_count = 0
    
    for i, entry in enumerate(qa_data):
        print(f"处理第 {i+1}/{len(qa_data)} 个样本: {entry['image']}")
        
        # 构建图片路径
        image_path = os.path.join(images_dir, entry['image'])
        
        # 调用API获取模型回答
        model_answer = call_api_for_question(client, image_path, entry['question'])
        
        # 比较答案
        is_correct = compare_answers(model_answer, entry['answer'])
        if is_correct:
            correct_count += 1
        
        # 记录结果
        result = {
            'image': entry['image'],
            'question': entry['question'],
            'standard_answer': entry['answer'],
            'model_answer': model_answer,
            'is_correct': is_correct
        }
        results.append(result)
        
        print(f"问题: {entry['question']}")
        print(f"标准答案: {entry['answer']}")
        print(f"模型回答: {model_answer}")
        print(f"正确性: {'✓' if is_correct else '✗'}")
        print("-" * 50)
        
        time.sleep(1)  # 避免API调用过快导致限制
    
    # 计算准确率
    accuracy = correct_count / len(qa_data) if qa_data else 0
    print(f"\n测试完成！")
    print(f"总样本数: {len(qa_data)}")
    print(f"正确样本数: {correct_count}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return results, accuracy

def save_results(results: List[Dict], accuracy: float, output_file: str):
    """保存测试结果到文件"""
    output_data = {
        'accuracy': accuracy,
        'total_samples': len(results),
        'correct_samples': sum(1 for r in results if r['is_correct']),
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")

def main():
    # 配置路径
    qa_file_path = "C:\\E\\CQA-survey\\A_Data_TEST\\data\\Q_and_A.txt"
    images_dir = "C:\\E\\CQA-survey\\A_Data_TEST\\data\\org_chart_images"
    output_file = "C:\\E\\CQA-survey\\A_Data_TEST\\results\\results_Pro-GLM-4.1V-9B-Thinking.json"

    # 测试模型
    results, accuracy = test_model_on_dataset(qa_file_path, images_dir)
    
    # 保存结果
    save_results(results, accuracy, output_file)

if __name__ == "__main__":
    main()