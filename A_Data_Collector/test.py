import os

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
    counter = 1

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

if __name__ == '__main__':

    target_folder = 'C:\\E\\CQA-Dataset\\A_Data_Collector\\org_chart_images_from_urls_rename'

    rename_images_in_folder(target_folder)