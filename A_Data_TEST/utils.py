import json

def remove_api_errors(input_file, output_file):
    """
    删除JSON文件中所有API调用错误的条目并保存到新文件
    
    参数:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    try:
        # 读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 过滤掉API调用错误的条目
        filtered_results = [
            entry for entry in data['results'] 
            if not (isinstance(entry.get('model_answer', ''), str) 
                   and entry['model_answer'].startswith('API调用错误:'))
        ]
        
        # 更新统计数据
        new_data = {
            "accuracy": len([r for r in filtered_results if r['is_correct']]) / len(filtered_results),
            "total_samples": len(filtered_results),
            "correct_samples": len([r for r in filtered_results if r['is_correct']]),
            "results": filtered_results
        }
        
        # 写入新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成！已保存到 {output_file}")
        print(f"原始条目数: {data['total_samples']}, 处理后条目数: {new_data['total_samples']}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    input_json = "C:\\E\\CQA-survey\\A_Data_TEST\\results\\results_QwenVL-78B.json"  # 替换为你的输入文件路径
    output_json = "C:\\E\\CQA-survey\\A_Data_TEST\\results\\results_QwenVL-78B_processed.json"  # 替换为你想要的输出文件路径
    remove_api_errors(input_json, output_json)