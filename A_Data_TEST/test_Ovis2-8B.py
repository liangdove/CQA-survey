import os
# 设置 Hugging Face 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Remove OpenAI import and add transformers imports
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import base64
import json
from PIL import Image
import io
import os
from tqdm import tqdm
import argparse
import time
from typing import List, Dict, Tuple

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

def load_ovis_model(model_path="AIDC-AI/Ovis1.6-Gemma2-9B", device="cuda"):
    print(f"Loading Ovis model from {model_path}...")
    
    # 加载模型和处理器，添加 use_fast=True 来避免警告
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print("Model loaded successfully!")
    return model, processor

def inference_with_ovis(model, processor, image, query, device="cuda"):
    """模型推理，使用Ovis模型的正确接口，参考https://huggingface.co/AIDC-AI/Ovis2-8B"""
    
    # 根据 test.py 的格式准备输入
    text = f"{query}回答时仅用一个词或短语，如：“数字”、“职位名称”、“第三层/第二层/第一层”、“是/否”、“姓名”等简洁词汇回答问题。"
    query_text = f'<image>\n{text}'
    
    try:
        # 使用 Ovis2 样例中的预处理方式
        max_partition = 9  # 根据样例代码设置
        images = [image]  # 将单个图像包装成列表
        
        # 使用 model.preprocess_inputs 方法（如果存在）
        if hasattr(model, 'preprocess_inputs'):
            # 必须是格式化的对话模板！！preprocess_inputs方法给定了
            prompt, input_ids, pixel_values = model.preprocess_inputs(query_text, images, max_partition=max_partition)
            text_tokenizer = model.get_text_tokenizer() # ovis 方法，不是hf标准做法
            visual_tokenizer = model.get_visual_tokenizer()
            
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
            
            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
            pixel_values = [pixel_values]  # 图像emb也是列表
            
            # 生成输出
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=50, # 50即可
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                response = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                
                # 提取生成的部分（去掉输入提示）
                if prompt in response:
                    response = response.replace(prompt, "").strip()
                
                return response
        else:
            # 如果没有 preprocess_inputs 方法，使用传统方式
            # 但避免使用 apply_chat_template，直接处理文本和图像
            inputs = processor(text=text, images=image, return_tensors="pt")
            
            # 移动到指定设备
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None and hasattr(v, 'to')}
            
            # 生成回复
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            # 解码回复（只取新生成的部分）
            generated_tokens = output[0][inputs['input_ids'].shape[1]:]
            response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
    except Exception as e:
        print(f"Error in Ovis inference: {e}")
        print(f"Query: {query}")
        print(f"Image type: {type(image)}")
        
        # 最后的备用方案：返回默认值
        return "Error"

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
    model_path: str = "AIDC-AI/Ovis1.6-Gemma2-9B"
):
    """在自定义VQA数据集上评估模型"""
    
    # Initialize local Ovis model instead of OpenAI client
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_ovis_model(model_path, device)
    
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
            
            # 构建图片路径并加载图片
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                model_answer = "图片文件不存在"
                is_correct = False
            else:
                # 加载图片
                image = Image.open(image_path).convert('RGB')
                
                # 使用模型推理
                model_answer = inference_with_ovis(model, processor, image, question, device)
                
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
            if (i + 1) % 10 == 0:
                current_acc = correct_count / total_count
                print(f"已处理 {i+1}/{len(qa_data)}, 当前准确率: {current_acc:.4f}")
            
            # 避免GPU内存积累
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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
    
    # 保存结果 - 与test.py保持一致的格式
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
    parser = argparse.ArgumentParser(description='使用HuggingFace模型在自定义VQA数据集上评估')
    parser.add_argument('--qa_file', type=str, 
                       default="C:\\E\\CQA-survey\\A_Data_TEST\\data\\Q_and_A.txt",
                       help='Q&A数据集文件路径')
    parser.add_argument('--images_dir', type=str, 
                       default="C:\\E\\CQA-survey\\A_Data_TEST\\data\\org_chart_images",
                       help='图片文件夹路径')
    parser.add_argument('--output', type=str, 
                       default='hf_vqa_results.json', 
                       help='输出结果文件')
    parser.add_argument('--limit', type=int, default=None, 
                       help='限制测试样本数量')
    parser.add_argument('--model_path', type=str, default='AIDC-AI/Ovis1.6-Gemma2-9B',
                       help='HuggingFace模型路径或本地模型路径')
    
    args = parser.parse_args()
    
    # 运行评估
    evaluate_vqa_on_custom_dataset(
        qa_file_path=args.qa_file,
        images_dir=args.images_dir,
        output_file=args.output,
        limit=args.limit,
        model_path=args.model_path
    )

# 使用示例:
# python chartqa_test_hf_infer.py --qa_file "C:\E\CQA-survey\A_Data_TEST\data\Q_and_A.txt" --images_dir "C:\E\CQA-survey\A_Data_TEST\data\org_chart_images" --output "hf_results.json"

# 测试少量样本:
# python chartqa_test_hf_infer.py --limit 10 --output "hf_test_results.json"

# 使用不同模型:
# python chartqa_test_hf_infer.py --model_path "AIDC-AI/Ovis2-8B" --output "ovis2_results.json"
# 数据集加载：使用datasets库加载ChartQA数据集
# 批量处理：支持整个数据集的批量评测
# 评测指标：计算准确率并支持多个正确答案
# 结果保存：将详细结果保存为JSON文件
# 进度显示：使用tqdm显示评测进度
# 错误处理：添加异常处理避免单个样本错误影响整体评测
# 命令行参数：支持灵活配置评测参数 
# """

# # # 安装依赖
# # pip install -r requirements.txt

# # # 直接运行评测，会自动下载数据集
# # python infer.py --split test --limit 10

