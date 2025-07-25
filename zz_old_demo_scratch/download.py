from datasets import load_dataset

import requests
try:
    response = requests.get("https://hf-mirror.com", timeout=5)
    print("网络连通正常" if response.status_code == 200 else "服务器响应异常")
except Exception as e:
    print(f"网络连接失败: {str(e)}")


from datasets import load_dataset
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载数据集并指定缓存目录
dataset = load_dataset(
    "liuhaotian/LLaVA-CC3M-Pretrain-595K",
    cache_dir="C:\\E\\mllm\\dataset"
)

# 保存为磁盘文件到目标目录
dataset.save_to_disk("C:\\E\\mllm\\local_dataset")

print("✅ 数据集已保存到 C:\\E\\mllm\\local_dataset")
