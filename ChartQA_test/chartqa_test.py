import os
# 设置 Hugging Face 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from openai import OpenAI
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
        # Load from local path
        print(f"Loading dataset from local path: {dataset_path}")
        dataset = load_dataset(dataset_path, split=split)
    else:
        # Try to load from cache or download
        print("Loading ChartQA dataset...")
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
    if isinstance(image, str):
        # If it's a file path
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # If it's a PIL Image
        buffer = io.BytesIO()
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            # Create a white background
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            image = rgb_image
        elif image.mode not in ['RGB', 'L']:
            # Convert other modes to RGB
            image = image.convert('RGB')
            
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def evaluate_vqa(dataset_path=None, split='test', output_file='results.json', limit=None, cache_dir="./chartqa_data"):
    """Evaluate VQA model on ChartQA dataset"""
    
    # Initialize OpenAI client
    client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
    model_name = client.models.list().data[0].id
    
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
            
            # Prepare the prompt
            text = f"Analyze the chart and provide the answer to the following question with just one word or number: {query}"
            
            # Make API call
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
                            'url': f"data:image/jpeg;base64,{image_to_base64(image)}"
                        },
                    }],
                }],
                temperature=0.1,  # Lower temperature for more consistent results
                top_p=0.9
            )
            
            # Extract the answer
            predicted_answer = response.choices[0].message.content.strip()
            
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
            
            # Print progress every 10 samples
            if (i + 1) % 100 == 0:
                current_acc = correct / total
                print(f"Processed {i+1}/{len(dataset)}, Accuracy: {current_acc:.4f}")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0
    
    # Save results
    evaluation_results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model': model_name,
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
            cache_dir="/data/coding/chartqa_data/HuggingFaceM4___chart_qa"
        )

# # 安装依赖
# pip install -r requirements.txt

# # 运行完整测试集评测
# python infer.py --split test --output test_results.json

# # 运行验证集评测（限制100个样本）
# python infer.py --split val --limit 100 --output val_results.json

# # 指定数据集路径
# python infer.py --dataset_path /path/to/chartqa --split test

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


