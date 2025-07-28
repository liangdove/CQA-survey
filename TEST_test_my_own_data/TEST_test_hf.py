import torch
from PIL import Image
from transformers import AutoModelForCausalLM
import os
import re

# Ovis2参照了huggingface模型卡片，https://modelscope.cn/models/AIDC-AI/Ovis2-8B

# 首先加载模型，这里使用ms上的Ovis2-8B模型
model = AutoModelForCausalLM.from_pretrained("/data/coding/Model/Ovis2-8B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# 定义一个函数将图片转换为base64编码
def parse_qa_file(qa_file_path):
    qa_pairs = []
    
    with open(qa_file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    # print(content)

    groups = content.split('\n\n')
    
    for group in groups:
        lines = group.strip().split('\n')
        if len(lines) >= 3:
            image_id = lines[0].strip()
            question = lines[1].strip() # 处理问答对
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
    """处理视觉问答任务"""
    qa_pairs = parse_qa_file(qa_file_path)
    
    results = [] # 结果
    
    for i, qa in enumerate(qa_pairs):
        image_id = qa['image_id']
        question = qa['question']
        expected_answer = qa['expected_answer']
        
        # 构建图片路径
        image_path = os.path.join(image_folder, f"{image_id}.jpg")
        
        print(f"处理第 {i+1}/{len(qa_pairs)} 个问题...")
        print(f"图片ID: {image_id}")
        print(f"问题: {question}")
        
        try:
            # 加载图片
            images = [Image.open(image_path)]
            max_partition = 9
            
            # 构建提示文本
            text = f"{question}"
            query = f'<image>\n{text}'
            
            # 预处理输入
            prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
            pixel_values = [pixel_values]
            
            # 生成回答
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=1024,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None, # 默认0.1
                    repetition_penalty=None,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                api_answer = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            
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
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"图片ID: {result['image_id']}\n")
            f.write(f"问题: {result['question']}\n")
            f.write(f"预期答案: {result['expected_answer']}\n")
            f.write(f"API回答: {result['api_answer']}\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    image_folder = "/data/coding/TEST_data/image"
    qa_file_path = "/data/coding/TEST_data/Q&A.txt"
    output_file_1 = "/data/coding/batch_results_1.txt"
    output_file_2 = "/data/coding/batch_results_2.txt"
    output_file_3 = "/data/coding/batch_results_3.txt"

    print("开始批量视觉问答任务...")
    results = visual_qa(image_folder, qa_file_path)
    
    print(f"\n任务完成！共处理 {len(results)} 个问答对")
    
    # 保存结果
    save_results(results, output_file_3)
    print(f"结果已保存到: {output_file_3}")