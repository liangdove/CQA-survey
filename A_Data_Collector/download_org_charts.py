import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random

# --- 配置区 ---
# ChromeDriver 的路径
# 如果你把 chromedriver.exe 放在了脚本同目录下，可以写成 DRIVER_PATH = './chromedriver.exe'
# 在新版 Selenium (4.6+) 中，如果 chromedriver 在系统 PATH 中，可以不指定路径
DRIVER_PATH = 'C:\\E\\CQA-Dataset\\A_Data_Collector\\chromedriver.exe' 
# Chrome浏览器的路径（添加这行）
CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
# 公司名单文件
COMPANIES_FILE = 'C:\\E\\CQA-Dataset\\A_Data_Collector\\companies.txt'
# 图片保存的主文件夹
OUTPUT_DIR = 'C:\\E\\CQA-Dataset\\A_Data_Collector\\org_chart_images'
# 为每个公司下载的图片数量 (-1 表示下载所有找到的图片)
IMAGES_TO_DOWNLOAD = 20
# 最大图片数量限制 (防止下载过多图片)
MAX_IMAGES_PER_COMPANY = 20
# --- 配置区结束 ---

def setup_driver():
    """配置并启动 Selenium WebDriver"""
    service = Service(executable_path=DRIVER_PATH)
    options = webdriver.ChromeOptions()
    options.binary_location = CHROME_PATH
    # options.add_argument('--headless')  # 无头模式，不在屏幕上显示浏览器窗口
    options.add_argument('--log-level=3') # 减少控制台不必要的日志
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# def download_image(url, folder_path, file_name):
#     """下载单张图片"""
#     try:
#         response = requests.get(url, stream=True, timeout=10)
#         if response.status_code == 200:
#             # 从URL猜测文件扩展名，如果没有则默认为 .jpg
#             extension = os.path.splitext(url.split('?')[0])[-1] or '.jpg'
#             if not extension.startswith('.'):
#                 extension = '.jpg' # 兜底处理
            
#             with open(os.path.join(folder_path, f"{file_name}{extension}"), 'wb') as f:
#                 for chunk in response.iter_content(1024):
#                     f.write(chunk)
#             print(f"  成功下载图片: {file_name}{extension}")
#         else:
#             print(f"  下载失败，状态码: {response.status_code}, URL: {url}")
#     except Exception as e:
#         print(f"  下载图片时发生错误: {e}")

def download_image(url, folder_path, file_name):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://image.baidu.com/',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        if response.status_code == 200:
            extension = os.path.splitext(url.split('?')[0])[-1] or '.jpg'
            if not extension.startswith('.'):
                extension = '.jpg'
            path = os.path.join(folder_path, f"{file_name}{extension}")
            with open(path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"  下载成功: {file_name}{extension}")
            return True
        else:
            print(f"  下载失败: 状态码 {response.status_code}")
    except Exception as e:
        print(f"  下载失败: {str(e)}")
    return False

def search_and_download(driver, company_name):
    """为单个公司执行搜索和下载"""
    print(f"\n正在处理公司: {company_name}")
    
    # 1. 创建公司专属文件夹
    company_folder = os.path.join(OUTPUT_DIR, company_name)
    os.makedirs(company_folder, exist_ok=True)

    # 2. 访问百度图片并搜索
    try:
        driver.get("https://image.baidu.com")
        print("  正在加载百度图片页面...")
        
        # 等待页面完全加载
        wait = WebDriverWait(driver, 15)
        
        # 尝试多种方式定位搜索框
        search_box = None
        search_selectors = [
            (By.NAME, "word"),
        ]
        
        for selector_type, selector_value in search_selectors:
            try:
                search_box = wait.until(EC.presence_of_element_located((selector_type, selector_value)))
                print(f"  成功找到搜索框: {selector_type} = {selector_value}")
                break
            except:
                continue
        
        if not search_box:
            print(f"  错误: 无法找到搜索框，跳过 {company_name}")
            return
        
        # 清空搜索框并输入搜索词
        search_box.clear()
        search_query = f"{company_name}公司组织架构图"
        search_box.send_keys(search_query)
        print(f"  搜索关键词: {search_query}")
        search_box.send_keys(Keys.RETURN)

        # 3. 等待图片结果加载
        print("  正在等待搜索结果加载...")
        time.sleep(5)
        
        # 获取图片容器
        print("  正在获取图片容器...")
        img_containers = []
        container_selectors = [
            '[data-objurl]',
        ]
        
        for selector in container_selectors:
            try:
                img_containers = driver.find_elements(By.CSS_SELECTOR, selector)
                if img_containers:
                    break
            except:
                continue
        
        if not img_containers:
            print(f"  警告: 未找到图片容器，跳过 {company_name}")
            return

        # 4. 下载图片
        count = 0
        total_images = len(img_containers)
        print(f"  找到 {total_images} 个图片容器，开始下载...")
        
        # 确定实际下载数量
        download_count = min(IMAGES_TO_DOWNLOAD, total_images) if IMAGES_TO_DOWNLOAD > 0 else min(MAX_IMAGES_PER_COMPANY, total_images)
        
        print(f"  计划下载 {download_count} 张图片")
        
        for i, container in enumerate(img_containers):
            if count >= download_count:
                break
            
            try:
                # 滚动到图片位置确保可见
                driver.execute_script("arguments[0].scrollIntoView(true);", container)
                time.sleep(0.5 + random.uniform(0, 0.5))
                
                # 获取原图URL
                img_url = None
                for attr in ['data-objurl', 'data-src', 'data-original']:
                    img_url = container.get_attribute(attr)
                    if img_url and img_url.startswith('http'):
                        break
                
                # 验证URL并下载
                if img_url and img_url.startswith('http') and 'base64' not in img_url and len(img_url) > 50:
                    print(f"  正在下载第 {count + 1}/{download_count} 张图片...")
                    print(f"    图片URL: {img_url[:100]}...")
                    if download_image(img_url, company_folder, f"{company_name}_{count + 1}"):
                        count += 1
                    time.sleep(1 + random.uniform(0, 1))  # 增加延时避免被检测
                else:
                    print(f"  跳过第 {i+1} 张无效图片URL: {img_url[:50] if img_url else 'None'}...")
                    
            except Exception as e:
                print(f"  处理第 {i+1} 张图片时出错: {e}")
                continue
        
        print(f"  完成 {company_name}，成功下载 {count}/{download_count} 张图片")

    except Exception as e:
        print(f"处理 {company_name} 时出错: {e}")
        # 保存页面源码以便调试
        try:
            with open(f"debug_{company_name}.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print(f"  已保存调试页面: debug_{company_name}.html")
        except:
            pass


def main():
    """主函数"""
    if not os.path.exists(COMPANIES_FILE):
        print(f"错误: 公司名单文件 '{COMPANIES_FILE}' 不存在。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(COMPANIES_FILE, 'r', encoding='utf-8') as f:
        companies = [line.strip() for line in f if line.strip()]

    print(f"共找到 {len(companies)} 个公司")
    print(f"图片下载设置: {'下载所有图片' if IMAGES_TO_DOWNLOAD == -1 else f'每个公司下载 {IMAGES_TO_DOWNLOAD} 张'}")
    print(f"最大下载限制: {MAX_IMAGES_PER_COMPANY} 张/公司")

    driver = setup_driver()
    try:
        total_downloaded = 0
        for idx, company in enumerate(companies, 1):
            print(f"\n[{idx}/{len(companies)}] 开始处理公司: {company}")
            search_and_download(driver, company)
            
            # 统计已下载的图片数量
            company_folder = os.path.join(OUTPUT_DIR, company)
            if os.path.exists(company_folder):
                company_images = len([f for f in os.listdir(company_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
                total_downloaded += company_images
            
            # 在处理不同公司之间加入较长延时，这是非常重要的反-反爬虫策略
            if idx < len(companies):  # 不是最后一个公司
                print(f"...等待10秒后处理下一个公司... (已完成 {idx}/{len(companies)})")
                time.sleep(10)  # 减少等待时间以提高效率
                
    except KeyboardInterrupt:
        print("\n用户中断操作")
    finally:
        driver.quit()
        print(f"\n所有任务完成。总共下载了 {total_downloaded} 张图片。")

if __name__ == '__main__':
    main()