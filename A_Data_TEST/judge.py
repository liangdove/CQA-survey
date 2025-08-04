# 裁判模型
# 用于判断json结果文件中的回答是否语义上是正确的，如果是正确的，就将其is_correct标记为正确，反之则标记为错误

import json
import time
from openai import OpenAI
from typing import Dict, List

def create_judge_client():
    """创建裁判模型客户端"""
    return OpenAI(
        api_key="xxx", 
        base_url="https://api.siliconflow.cn/v1"
    )

def create_judge_prompt(question: str, standard_answer: str, model_answer: str) -> str:
    """创建裁判模型的提示词"""
    prompt = f"""你是一个专业的答案评判助手，需要判断模型回答是否与标准答案在语义上一致。

请仔细分析以下内容：
- 问题：{question}
- 标准答案：{standard_answer}
- 模型回答：{model_answer}

评判规则：
1. 数字一致性：
   - "7" 和 "七个"、"7个"、"七" 应视为相同
   - "2" 和 "两个"、"2个"、"二" 应视为相同
   - "第三层" 和 "第3层"、"3层" 应视为相同
   - 数字不匹配应视为错误

2. 语义一致性：
   - "是" 和 "正确"、"对" 应视为相同
   - "否" 和 "不是"、"错误"、"不对" 应视为相同

3. 内容包含性：
   - 如果模型回答包含了标准答案的核心信息，即使有额外描述也算正确
   - 例如："总监理工程师代表下有七个直接下属部门" 包含了 "7" 的信息

4. 特殊情况：
   - 如果模型回答是"API调用错误"或"图片文件不存在"等错误信息，直接判为错误
   - 如果回答完全偏离主题或答非所问，判为错误
   - 如果模型回答的标点符号与标准答案不一致，但语义相同，仍然视为正确

请只回答 "正确" 或 "错误"，不要包含其他内容。"""
    
    return prompt

def judge_single_answer(client, question: str, standard_answer: str, model_answer: str) -> bool:
    """使用裁判模型判断单个答案的正确性"""
    try:
        prompt = create_judge_prompt(question, standard_answer, model_answer)
        
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=10,
            temperature=0.1  # 降低温度以获得更一致的判断
        )
        
        judgment = response.choices[0].message.content.strip()
        
        # 解析判断结果
        if "正确" in judgment:
            return True
        elif "错误" in judgment:
            return False
        else:
            # 如果回答不明确，使用备用逻辑
            print(f"警告：裁判模型回答不明确: {judgment}")
            return False
            
    except Exception as e:
        print(f"裁判模型调用错误: {str(e)}")
        return False

def judge_all_results(input_file: str, output_file: str):
    """对所有测试结果进行重新判断"""
    # 读取测试结果
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    client = create_judge_client()
    results = data['results']
    
    print(f"开始对 {len(results)} 个结果进行重新判断...")
    
    # 统计信息
    original_correct = data['correct_samples']
    new_correct_count = 0
    changed_count = 0
    
    start_time = time.time()
    for i, result in enumerate(results):
        print(f"处理第 {i+1}/{len(results)} 个结果...")
        
        # 获取原始信息
        question = result['question']
        standard_answer = result['standard_answer']
        model_answer = result['model_answer']
        original_judgment = result['is_correct']
        
        # 使用裁判模型重新判断
        new_judgment = judge_single_answer(client, question, standard_answer, model_answer)
        
        # 更新结果
        result['is_correct'] = new_judgment
        result['original_judgment'] = original_judgment  # 保存原始判断
        
        if new_judgment:
            new_correct_count += 1
        
        if new_judgment != original_judgment:
            changed_count += 1
            print(f"判断发生变化: {original_judgment} -> {new_judgment}")
            print(f"问题: {question}")
            print(f"标准答案: {standard_answer}")
            print(f"模型回答: {model_answer}")
            print("-" * 50)
        
        # 避免API调用过快
        time.sleep(0.2)
    
    # 更新准确率
    new_accuracy = new_correct_count / len(results) if results else 0
    data['accuracy'] = new_accuracy
    data['correct_samples'] = new_correct_count
    
    # 添加判断统计信息
    data['judge_stats'] = {
        'original_correct': original_correct,
        'original_accuracy': original_correct / len(results) if results else 0,
        'new_correct': new_correct_count,
        'new_accuracy': new_accuracy,
        'changed_count': changed_count,
        'improvement': new_correct_count - original_correct
    }
    
    end_time = time.time()
    print("耗时：", end_time - start_time, "秒")
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n重新判断完成！")
    print(f"原始正确数: {original_correct}")
    print(f"重判正确数: {new_correct_count}")
    print(f"判断变化数: {changed_count}")
    print(f"准确率提升: {new_accuracy - data['judge_stats']['original_accuracy']:.4f}")
    print(f"新准确率: {new_accuracy:.4f} ({new_accuracy*100:.2f}%)")
    print(f"结果已保存到: {output_file}")

# def batch_judge_multiple_files(file_list: List[str]):
#     """批量处理多个测试结果文件"""
#     for input_file in file_list:
#         print(f"\n处理文件: {input_file}")
#         output_file = input_file.replace('.json', '_judged.json')
#         judge_all_results(input_file, output_file)

def main():
    # 单个文件处理
    input_file = "C:\\E\\CQA-survey\\A_Data_TEST\\results\\results_Ovis2-8B.json"
    output_file = "C:\\E\\CQA-survey\\A_Data_TEST\\results\\results_Ovis2-8B_judged.json"

    judge_all_results(input_file, output_file)
    
    # 如果需要批量处理多个文件，可以使用：
    # file_list = [
    #     "C:\\E\\CQA-survey\\A_Data_TEST\\test_results1.json",
    #     "C:\\E\\CQA-survey\\A_Data_TEST\\test_results2.json"
    # ]
    # batch_judge_multiple_files(file_list)

if __name__ == "__main__":
    main()