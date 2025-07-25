## CQA-survey


## 模型：

![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507211655985.png)

#### 方法一：

![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507211652051.png)

通过深度学习方法先识别图表，变成文本，然后加上问题，一起送给大语言模型完成问答。

找到合适的DataExtracter: 提取图表数据转化为模型可以理解的结构化数据。几种处理方案：

ChartT5:  
https://github.com/zmykevin/ACL2023_ChartT5

PlotQA:  
https://github.com/NiteshMethani/PlotQA

UniChart:  
https://github.com/vis-nlp/UniChart

#### 方法二：
![](https://cdn.jsdelivr.net/gh/l
iangdove/PicGo/imgs/202507242232652.png)

端到端融合。

考虑到以后要用数据集做微调，事先运行了一个[demo](https://github.com/liangdove/CQA-survey/tree/main/src)。采用的是<|image|> token预留策略。

## 数据集：
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507242230426.png)

## 参考资料
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507251658096.png)

InternVL使用的数据集：  
https://zhuanlan.zhihu.com/p/703940563




## 测试

测试数据集分为自定义数据集测试和ChartQA标准数据集
| 数据集             | 模型版本           | 准确率      | 模型简介                           | 备注         |
|------------------|-------------------|------------------------|------------------------------------|--------------|
| test_data（自定义） | InternVL3-8B       | 0.714                  | 8B参数量，基础多模态模型           |              |
| test_data（自定义） | InternVL3-14B-4bit | xx                     | 14B参数量，4bit量化多模态模型       | xx     |
| ChartQA（标准）    | InternVL3-8B       | 0.7568 (1892/2500)     | 8B参数量，基础多模态模型           |              |
| ChartQA（标准）    | InternVL3-14B-4bit | xx                     | 14B参数量，4bit量化多模态模型       | xx     |



