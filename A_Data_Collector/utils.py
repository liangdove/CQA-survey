import os
from collections import defaultdict
from PIL import Image

def count_image_files(directory_path):
    """
    统计指定文件夹中的图片文件数量。

    参数:
    directory_path (str): 要统计的图片文件夹路径。

    返回:
    None: 直接打印统计结果。
    """
    if not os.path.isdir(directory_path):
        print(f"错误: 目录 '{directory_path}' 不存在。")
        return

    # 定义要专门统计的图片格式
    specific_formats = {'gif', 'jpg', 'png', 'jpeg'}
    
    # 使用 defaultdict 初始化计数器
    counts = defaultdict(int)
    total_count = 0

    print(f"正在统计目录 '{directory_path}' 中的图片...")

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 构造完整的文件路径
        file_path = os.path.join(directory_path, filename)

        # 检查是否为文件
        if os.path.isfile(file_path):
            # 获取文件扩展名并转换为小写
            try:
                extension = filename.split('.')[-1].lower()
                if extension == filename: # 处理没有扩展名的文件
                    continue
            except IndexError:
                continue # 文件名中没有 '.', 跳过

            # 检查是否是已知的图片格式
            # 这里可以根据需要添加更多图片格式
            all_image_formats = {'gif', 'jpg', 'png', 'jpeg', 'bmp', 'tiff', 'webp', 'svg', 'ico'}
            
            if extension in all_image_formats:
                total_count += 1
                if extension in specific_formats:
                    counts[extension] += 1
                else:
                    counts['other'] += 1

    # 打印结果
    print("\n--- 统计结果 ---")
    print(f"总图片数量: {total_count}")
    print("-" * 20)
    for fmt in sorted(specific_formats):
        print(f"{fmt.upper()} 文件数量: {counts[fmt]}")
    print(f"其他格式图片数量: {counts['other']}")
    print("--- 统计完成 ---\n")
    
def convert_images_to_png(directory_path):
    """
    将指定文件夹中所有非 JPG、PNG 格式的图片文件转换为 PNG 格式, 并删除原文件。

    参数:
    directory_path (str): 包含图片文件的文件夹路径。

    返回:
    None: 直接打印转换过程。
    """
    if not os.path.isdir(directory_path):
        print(f"错误: 目录 '{directory_path}' 不存在。")
        return

    # 定义不需要转换的格式
    formats_to_skip = {'.png', '.jpg', '.jpeg'}
    
    print(f"\n--- 开始转换图片到 PNG 格式 ---")
    print(f"目标目录: '{directory_path}'")

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 分离文件名和扩展名
        file_base, file_ext = os.path.splitext(filename)
        file_ext_lower = file_ext.lower()

        # 如果是不需要转换的格式，则跳过
        if file_ext_lower in formats_to_skip:
            continue

        original_file_path = os.path.join(directory_path, filename)
        
        # 确保是文件而不是目录
        if not os.path.isfile(original_file_path):
            continue

        # 定义新的PNG文件名和路径
        new_filename = f"{file_base}.png"
        new_file_path = os.path.join(directory_path, new_filename)

        try:
            # 打开图片文件
            with Image.open(original_file_path) as img:
                # 转换并保存为PNG格式
                img.save(new_file_path, 'PNG')
                print(f"成功: '{filename}' -> '{new_filename}'")
            
            # 删除原文件
            os.remove(original_file_path)
            print(f"删除原文件: '{filename}'")

        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            print(f"失败: 无法转换 '{filename}'. 错误: {e}")
    
    print("--- 图片转换完成 ---\n")
    
def rename_images_in_folder(folder_path):
    """
    将指定文件夹中的所有图片文件按数字顺序重命名（例如 1.jpg, 2.png）。

    参数:
    folder_path (str): 包含图片的文件夹路径。
    """
    # 支持的图片文件扩展名
    supported_extensions = ('.jpg', '.jpeg', '.png')

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return

    # 获取文件夹中所有文件的列表
    try:
        files = os.listdir(folder_path)
    except OSError as e:
        print(f"错误：无法访问文件夹 '{folder_path}': {e}")
        return
        
    # 过滤出图片文件
    image_files = [f for f in files if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(supported_extensions)]
    
    # 对文件进行排序以确保重命名顺序一致
    image_files.sort()

    # 初始化重命名计数器
    counter = 685

    print(f"在 '{folder_path}' 中找到 {len(image_files)} 个图片文件。开始重命名...")

    # 遍历并重命名每个图片文件
    for filename in image_files:
        # 分离文件名和扩展名
        _, file_extension = os.path.splitext(filename)

        # 构建新的文件名
        new_filename = f"{counter}{file_extension}"

        # 获取旧文件和新文件的完整路径
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        # 如果新文件名已存在，则跳过，避免覆盖
        if os.path.exists(new_filepath):
            print(f"警告：文件名 '{new_filename}' 已存在，跳过 '{filename}'。")
            continue

        # 执行重命名
        try:
            os.rename(old_filepath, new_filepath)
            print(f"已重命名: '{filename}' -> '{new_filename}'")
            counter += 1
        except OSError as e:
            print(f"错误：重命名 '{filename}' 失败: {e}")

    print("\n所有图片重命名完成。")


def rename_images_from_number(folder_path, num, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    """
    对指定文件夹中的图片文件从给定数字num开始重命名
    
    参数:
        folder_path (str): 图片文件夹路径
        num (int): 起始编号
        extensions (tuple): 要处理的图片文件扩展名
        
    返回:
        int: 下一个可用的编号
    """
    # 确保文件夹存在
    if not os.path.isdir(folder_path):
        raise ValueError(f"文件夹 {folder_path} 不存在")
    
    # 获取文件夹中所有图片文件
    image_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(extensions):
            image_files.append(file)
    
    # 按文件名排序
    image_files.sort()
    
    # 重命名文件
    for i, filename in enumerate(image_files, start=num):
        # 获取文件扩展名
        ext = os.path.splitext(filename)[1].lower()
        
        # 构造新文件名
        new_name = f"{i}{ext}"
        
        # 原始文件完整路径
        old_path = os.path.join(folder_path, filename)
        
        # 新文件完整路径
        new_path = os.path.join(folder_path, new_name)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_name}")
    
    return num + len(image_files)

import shutil

def flatten_directory(root_dir):
    """
    将root_dir下的所有子文件夹中的文件移动到root_dir中
    """
    # 遍历根目录下的所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # 跳过根目录本身
        if dirpath == root_dir:
            continue
            
        # 处理当前子文件夹中的所有文件
        for filename in filenames:
            src_path = os.path.join(dirpath, filename)
            dst_path = os.path.join(root_dir, filename)
            
            # 处理文件名冲突
            counter = 1
            while os.path.exists(dst_path):
                name, ext = os.path.splitext(filename)
                dst_path = os.path.join(root_dir, f"{name}_{counter}{ext}")
                counter += 1
                
            # 移动文件
            shutil.move(src_path, dst_path)
            print(f"移动: {src_path} -> {dst_path}")
        
        # 删除空文件夹
        try:
            os.rmdir(dirpath)
            print(f"删除空文件夹: {dirpath}")
        except OSError:
            print(f"无法删除文件夹: {dirpath} (可能不为空或包含系统文件)")

def add_suffix_to_images(folder_path, suffix="_tong", extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    """
    给图片文件夹中的所有图片文件名添加后缀
    
    参数:
        folder_path (str): 图片文件夹路径
        suffix (str): 要添加的后缀（默认为"_tong"）
        extensions (tuple): 要处理的图片文件扩展名
        
    返回:
        list: 重命名后的新文件名列表
    """
    # 确保文件夹存在
    if not os.path.isdir(folder_path):
        raise ValueError(f"文件夹 {folder_path} 不存在")
    
    renamed_files = []
    
    # 获取文件夹中所有图片文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            # 分割文件名和扩展名
            name, ext = os.path.splitext(filename)
            
            # 构造新文件名（在原文件名后添加后缀）
            new_name = f"{name}{suffix}{ext}"
            
            # 原始文件完整路径
            old_path = os.path.join(folder_path, filename)
            
            # 新文件完整路径
            new_path = os.path.join(folder_path, new_name)
            
            # 重命名文件
            os.rename(old_path, new_path)
            renamed_files.append(new_name)
            print(f"重命名: {filename} -> {new_name}")

if __name__ == "__main__":
    # 提示用户输入文件夹路径
    folder_path = 'C:\\E\\CQA-Dataset\\A_Data_Collector\\org_chart_images'
    count_image_files(folder_path)
    # convert_images_to_png(folder_path)
    # count_image_files(folder_path)
    # rename_images_in_folder(folder_path)
    # flatten_directory(folder_path)
    rename_images_from_number(folder_path, 685)
    
