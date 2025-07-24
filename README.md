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
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507242232652.png)

端到端融合。

考虑到以后要用数据集做微调，运行了一个demo。采用的是<|image|> token预留策略。

## 数据集：
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507242230426.png)



