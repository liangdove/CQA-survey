import os
# 设置 Hugging Face 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Remove OpenAI import and add transformers imports
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import base64
from datasets import load_dataset
import json
from PIL import Image
import io
import os
from tqdm import tqdm
import argparse
import os


def download_chartqa_dataset(cache_dir="/data/coding/chartqa_data/HuggingFaceM4___chart_qa"):
    """Download ChartQA dataset from Hugging Face Hub"""
    print("Downloading ChartQA dataset from Hugging Face Hub...")
    try:
        # Download the dataset - ChartQA is available on Hugging Face
        dataset = load_dataset("HuggingFaceM4/ChartQA", cache_dir=cache_dir)
        print("Dataset downloaded successfully!")
        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return None

def load_chartqa_dataset(dataset_path=None, split='test', cache_dir="./chartqa_data"):
    """Load ChartQA dataset with automatic download if needed"""
    
    if dataset_path and os.path.exists(dataset_path):
        # 加载本地路径的数据集
        dataset = load_dataset(dataset_path, split=split)
    else:
        # 尝试从缓存加载或下载
        try:
            dataset = load_dataset("HuggingFaceM4/ChartQA", split=split, cache_dir=cache_dir)
            print(f"Loaded {len(dataset)} samples from {split} split")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Attempting to download dataset...")
            full_dataset = download_chartqa_dataset(cache_dir)
            if full_dataset is None:
                raise Exception("Failed to download dataset")
            dataset = full_dataset[split]
    
    return dataset

def image_to_base64(image):
    """Convert PIL Image or image path to base64"""
    if isinstance(image, str):  # 如果是文件路径
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        buffer = io.BytesIO()
        
        #RGBA to RGB
        if image.mode == 'RGBA':
            # 如果是 RGBA 模式，转换为 RGB
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])  # 使用 alpha 通道作为掩码
            image = rgb_image
        elif image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
            
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

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
    
    # 根据 Ovis2 样例代码的格式准备输入
    text = f"Analyze the chart and provide the answer to the following question with just one word or number: {query}"
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

def evaluate_vqa(dataset_path=None, split='test', output_file='results.json', limit=None, cache_dir="./chartqa_data", model_path="AIDC-AI/Ovis1.6-Gemma2-9B"):
    """Evaluate VQA model on ChartQA dataset"""
    
    # Initialize local Ovis model instead of OpenAI client
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_ovis_model(model_path, device)
    
    # Load dataset with automatic download
    dataset = load_chartqa_dataset(dataset_path, split, cache_dir)
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    results = []
    correct = 0
    total = 0
    
    print(f"Evaluating on {len(dataset)} samples from {split} split...")
    
    for i, sample in enumerate(tqdm(dataset)):
        try:
            image = sample['image']
            query = sample['query']
            labels = sample['label'] if isinstance(sample['label'], list) else [sample['label']]
            
            # Use local model inference instead of API call
            predicted_answer = inference_with_ovis(model, processor, image, query, device)
            
            # Check if prediction matches any of the labels (case-insensitive)
            is_correct = any(
                predicted_answer.lower().strip() == label.lower().strip() 
                for label in labels
            )
            
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            result = {
                'index': i,
                'query': query,
                'labels': labels,
                'prediction': predicted_answer,
                'correct': is_correct,
                'human_or_machine': sample.get('human_or_machine', None)
            }
            results.append(result)
            
            # Print progress every 50 samples
            if (i + 1) % 50 == 0:
                current_acc = correct / total
                print(f"Processed {i+1}/{len(dataset)}, Accuracy: {current_acc:.4f}")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # 计算最终指标
    accuracy = correct / total if total > 0 else 0
    
    # Save
    evaluation_results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model': model_path,
        'split': split,
        'samples': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation completed!")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to: {output_file}")
    
    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate VQA model on ChartQA dataset')
    parser.add_argument('--dataset_path', type=str, default=None, 
                       help='Path to local dataset (if not provided, will download from HuggingFace)')
    parser.add_argument('--split', type=str, default='test', 
                       choices=['train', 'val', 'test'], 
                       help='Dataset split to evaluate on')
    parser.add_argument('--output', type=str, default='results.json', 
                       help='Output file for results')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit number of samples to evaluate')
    parser.add_argument('--cache_dir', type=str, default='./chartqa_data',
                       help='Directory to cache downloaded dataset')
    parser.add_argument('--model_path', type=str, default='AIDC-AI/Ovis1.6-Gemma2-9B',
                       help='Path to local Ovis model or HuggingFace model name')
    parser.add_argument('--download_only', action='store_true',
                       help='Only download the dataset without evaluation')
    
    args = parser.parse_args()
    
    if args.download_only:
        download_chartqa_dataset(args.cache_dir)
    else:
        evaluate_vqa(
            dataset_path="/data/coding/chartqa_data/HuggingFaceM4___chart_qa",
            split=args.split,
            output_file=args.output,
            limit=args.limit,
            cache_dir="/data/coding/chartqa_data/HuggingFaceM4___chart_qa",
            model_path=args.model_path
        )

# # 安装依赖
# pip install -r requirements.txt

# # 运行完整测试集评测
# python chartqa_test_lmdeploy_api.py --split test --output test_results.json

# # 运行验证集评测（限制100个样本）
# python chartqa_test_lmdeploy_api.py --split val --limit 100 --output val_results.json

# # 指定数据集路径
# python chartqa_test_lmdeploy_api.py --dataset_path /path/tconao/chartqa --split test

# # 使用本地模型进行评测
# python chartqa_test_hf_infer.py --split test --limit 10 --output test_results.json --model_path /path/to/local/Ovis2-8B

""" 
数据集加载：使用datasets库加载ChartQA数据集
批量处理：支持整个数据集的批量评测
评测指标：计算准确率并支持多个正确答案
结果保存：将详细结果保存为JSON文件
进度显示：使用tqdm显示评测进度
错误处理：添加异常处理避免单个样本错误影响整体评测
命令行参数：支持灵活配置评测参数 
"""

# # 安装依赖
# pip install -r requirements.txt

# # 直接运行评测，会自动下载数据集
# python infer.py --split test --limit 10

