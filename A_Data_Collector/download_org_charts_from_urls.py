import os
import requests
import time
import random
from urllib.parse import urlparse

# def download_image(url, folder_path, file_name, retry=3):
#     """下载单张图片"""
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#         'Referer': 'https://image.baidu.com/',
#     }
    
#     for attempt in range(retry):
#         try:
#             response = requests.get(url, headers=headers, stream=True, timeout=10)
#             if response.status_code == 200:
#                 # 从URL提取文件扩展名
#                 parsed = urlparse(url)
#                 filename = os.path.basename(parsed.path)
#                 if not filename:
#                     filename = f"{file_name}.jpg"  # 默认扩展名
                
#                 # 确保文件名安全
#                 safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '.', '_'))
                
#                 path = os.path.join(folder_path, safe_filename)
#                 with open(path, 'wb') as f:
#                     for chunk in response.iter_content(1024):
#                         f.write(chunk)
#                 print(f"  下载成功: {safe_filename}")
#                 return True
#             else:
#                 print(f"  尝试 {attempt + 1}/{retry}: 状态码 {response.status_code}")
#                 time.sleep(2 ** attempt)  # 指数退避
#         except Exception as e:
#             print(f"  尝试 {attempt + 1}/{retry}: {str(e)}")
#             time.sleep(1)
#     return False

def download_image(url, folder_path, file_name):
    """下载单张图片（仅尝试一次）"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://image.baidu.com/',
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        if response.status_code == 200:
            # 从URL提取文件扩展名
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename:
                filename = f"{file_name}.jpg"  # 默认扩展名
            
            # 确保文件名安全
            safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '.', '_'))
            
            path = os.path.join(folder_path, safe_filename)
            with open(path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"  下载成功: {safe_filename}")
            return True
        else:
            print(f"  下载失败: 状态码 {response.status_code}")
            return False
    except Exception as e:
        print(f"  下载失败: {str(e)}")
        return False

def batch_download(url_file, output_folder):
    """批量下载图片"""
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取URL文件
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"共找到 {len(urls)} 个图片URL")
    
    success_count = 0
    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}] 正在处理: {url[:80]}...")
        
        try:
            # 生成文件名（使用序号+URL最后部分）
            parsed = urlparse(url)
            filename = f"{idx}_{os.path.basename(parsed.path)}" or f"image_{idx}"
            
            if download_image(url, output_folder, filename):
                success_count += 1
            
            # 随机延迟避免被封
            time.sleep(1 + random.uniform(0, 2))
            
        except Exception as e:
            print(f"  处理URL时出错: {e}")
            continue
    
    print(f"\n完成！成功下载 {success_count}/{len(urls)} 张图片")

if __name__ == '__main__':
    # 配置参数
    URL_FILE = 'C:\\E\\CQA-Dataset\\A_Data_Collector\\data-objurls.txt'  # 你的URL文本文件路径
    OUTPUT_FOLDER = 'C:\\E\\CQA-Dataset\\A_Data_Collector\\org_chart_images_from_urls'  # 图片保存目录

    batch_download(URL_FILE, OUTPUT_FOLDER)