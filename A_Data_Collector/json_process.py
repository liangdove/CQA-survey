import json
import re

def extract_qa_from_json(json_file_path, output_txt_path):
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for item in data:
            # 提取图片名称
            image_name = item.get('image_name', '')
            
            # 提取response中的Q和A
            response = item.get('response', '')
            
            # 使用正则表达式提取Q和A
            q_match = re.search(r'Q:\s*(.+?)\n', response)
            a_match = re.search(r'A:\s*(.+?)(?:\n|$)', response)
            
            if q_match and a_match:
                question = q_match.group(1).strip()
                answer = a_match.group(1).strip()
                
                # 写入格式化输出
                f.write(f"I: {image_name}\n")
                f.write(f"Q: {question}\n")
                f.write(f"A: {answer}\n")
                f.write("\n")  # 添加空行分隔
            else:
                print(f"Warning{image_name}")

if __name__ == "__main__":
    # 使用示例
    json_file_path = "C:\\E\\CQA-Dataset\\A_Data_Collector\\batch_results.json"  # 替换为你的JSON文件路径
    output_txt_path = "C:\\E\\CQA-Dataset\\A_Data_Collector\\Q_and_A.txt"  # 输出文件路径

    extract_qa_from_json(json_file_path, output_txt_path)
    print(f"提取完成，结果已保存到 {output_txt_path}")
