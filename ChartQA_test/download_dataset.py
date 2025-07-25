from datasets import load_dataset
import argparse
import os

def download_chartqa(cache_dir="./chartqa_data", splits=['train', 'val', 'test']):
    """Download ChartQA dataset"""
    print("Downloading ChartQA dataset from HuggingFace Hub...")
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download all splits
        dataset = load_dataset("HuggingFaceM4/ChartQA", cache_dir=cache_dir)
        
        print("Dataset downloaded successfully!")
        print("\nDataset information:")
        for split in splits:
            if split in dataset:
                print(f"  {split}: {len(dataset[split])} samples")
        
        # Show sample data structure
        sample = dataset['train'][0]
        print(f"\nSample data structure:")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Image type: {type(sample['image'])}")
        print(f"  Query example: {sample['query'][:100]}...")
        print(f"  Label example: {sample['label']}")
        
        return dataset
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download ChartQA dataset')
    parser.add_argument('--cache_dir', type=str, default='./chartqa_data',
                       help='Directory to cache downloaded dataset')
    
    args = parser.parse_args()
    
    download_chartqa(args.cache_dir)
