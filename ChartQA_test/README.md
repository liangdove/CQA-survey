## ChartQA benchmark

没有在网上找到官方benchmark工具，所以自己写了一个测试脚本。  
chartqa_test.py调用了后台部署的InternVL3模型。

实验结果：0.7568 准确率；实际准确率比这个数值高，因为很多答案误判了，比如参考答案是0.03，模型输出3%