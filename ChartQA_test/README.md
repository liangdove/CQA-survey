## ChartQA benchmark

#### ChartQA 数据集介绍

ChartQA 是一个用于图表理解任务的数据集，旨在评估模型从图表中提取信息并回答相关问题的能力。数据集包含多种类型的图表（如柱状图、折线图、饼图等）以及与之相关的问答对，覆盖了多种问题类型，包括数值提取、趋势分析和比较等。该数据集为研究图表理解提供了一个标准化的基准，广泛用于评估视觉语言模型的性能。我们在benchmark中使用其测试集，包含了**2500**张图表及其问答。


#### 评测方法
没有在网上找到官方benchmark工具，所以自己写了测试脚本。

和TEST_test_my_data一样，对于INternVL3模型，使用lmdeploy后端，加速推理。对于Ovis2模型，由于lmdeploy不支持，所以对其huggingface的模型卡的样例代码做进一步修改，改成适配ChartQA的格式。

[InternVL3评测代码实现](./chartqa_test_lmdeploy_api.py)

[Ovis2评测代码实现](./chartqa_test_hf_infer.py)


#### 评测结果

| 模型 | 准确率 | 正确数/总数 |
|------|--------|-------------|
| InternVL3-14B-4bit | 77.36% | 1934/2500 |
| InternVL3-8B | 75.68% | 1892/2500 |
| Ovis2-8B | 63.68% | 1592/2500 |

**结果分析：**
- InternVL3-14B-4bit 模型表现最佳，准确率达到 77.36% 虽然量化有失精度，但14B的模型表现仍然比8B的好。  

- InternVL3-8B 模型位居第二，准确率为 75.68%

- Ovis2-8B 模型准确率为 63.68%，相比 InternVL3 系列模型有一定差距

- 总体来看，InternVL3 系列模型在 ChartQA 任务上表现优于 Ovis2 模型
