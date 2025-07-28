from openai import OpenAI
import base64
import os
import re

# 老师给的数据集评测代码，因为数据量很少，可以直接放在代码中处理，评测指标为准确率
# 这里评测的是InternVL3系列模型，使用了lmdeploy部署后的API进行视觉问答任务

# 读取本地图片并转换为base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
# print(f" {model_name}")


def parse_qa_file(qa_file_path):
    qa_pairs = []
    
    with open(qa_file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    
    # 按空行分割每组问答
    groups = content.split('\n\n')
    
    for group in groups:
        lines = group.strip().split('\n')
        if len(lines) >= 3:
            # print(f"处理问答对: {lines[0].strip()}")
            image_id = lines[0].strip()
            question = lines[1].strip()
            answer = lines[2].strip()
            
            # 提取问题内容
            if question.startswith('Q：') or question.startswith('Q:'):
                question = question[2:].strip()
            
            # 提取答案内容
            if answer.startswith('A：') or answer.startswith('A:'):
                answer = answer[2:].strip()
            
            qa_pairs.append({
                'image_id': image_id,
                'question': question,
                'expected_answer': answer
            })
    
    return qa_pairs

def visual_qa(image_folder, qa_file_path):
    qa_pairs = parse_qa_file(qa_file_path)
    # print(f"共找到 {len(qa_pairs)} 个问答对")
    results = []
    
    for i, qa in enumerate(qa_pairs):
        image_id = qa['image_id']
        question = qa['question']
        expected_answer = qa['expected_answer']
        
        # 构建图片路径
        image_path = os.path.join(image_folder, f"{image_id}.jpg")
        
        # if not os.path.exists(image_path):
        #     print(f"警告：图片 {image_path} 不存在，跳过...")
        #     continue
        
        print(f"处理第 {i+1}/{len(qa_pairs)} 个问题...")
        print(f"图片ID: {image_id}")
        print(f"问题: {question}")
        
        try:
            # 构建提示文本
            text = f"{question}"
            
            # 调用API
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
                            'url': f"data:image/jpeg;base64,{image_to_base64(image_path)}"
                        },
                    }],
                }],
                temperature=0.8,
                top_p=0.8
            )
            
            # 提取回答
            api_answer = response.choices[0].message.content.strip()
            
            # 保存结果
            result = {
                'image_id': image_id,
                'question': question,
                'expected_answer': expected_answer,
                'api_answer': api_answer
            }
            results.append(result)
            
            print(f"预期答案: {expected_answer}")
            print(f"API回答: {api_answer}")
            print("-" * 50)
            
        except Exception as e:
            print(f"处理图片 {image_id} 时出错: {str(e)}")
            continue
    
    return results

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"图片ID: {result['image_id']}\n")
            f.write(f"问题: {result['question']}\n")
            f.write(f"预期答案: {result['expected_answer']}\n")
            f.write(f"API回答: {result['api_answer']}\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    image_folder = "/data/coding/test_data/image"
    qa_file_path = "/data/coding/test_data/Q&A.txt"
    output_file_1 = "/data/coding/batch_results_1.txt"
    output_file_2 = "/data/coding/batch_results_2.txt"
    output_file_3 = "/data/coding/batch_results_3.txt"

    print("开始批量视觉问答任务...")
    results = visual_qa(image_folder, qa_file_path)
    
    print(f"\n任务完成！共处理 {len(results)} 个问答对")
    
    # 保存结果
    save_results(results, output_file_3)
    print(f"结果已保存到: {output_file_3}")