# CQA-survey

<details>
<summary>综述文章调研</summary>

## 主流方法

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

考虑到以后要用数据集做微调，事先运行了一个[demo](https://github.com/liangdove/CQA-survey/tree/main/src)。采用的是<|image|> token预留策略。

## CQA开源数据集：
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507242230426.png)

ChartQA是22年发布的，并且有完整的训练集、发展集、测试集，比较适合做评测

</details>

<details>
<summary>模型调研</summary>

因为自建数据集（老师给定的数据）是中文图表，而ChartT5、UniChart都是用英文图表数据训练的，所以效果预测会很差。

应该选用中英双语料的模型（Qwen系列），探索一些开源模型。

#### InternVL系列论文：
https://arxiv.org/abs/2504.10479
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507251658096.png)

![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507261615061.png)

![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507281642237.png)

#### IDP排行榜
https://idp-leaderboard.org/
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507261646631.png)

#### OpenCompass排行榜
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507261949249.png)


#### InternVL使用的数据集：  
https://zhuanlan.zhihu.com/p/703940563
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507261649192.png)

</details>


---

## 总体流程

**同步仓库**: https://github.com/liangdove/CQA-survey

## 数据选择

#### 图表数据集

- 老师给定的数据（自建数据集）

- ChartQA数据集

CQA目前开放的数据集比较少，具有代表性的是**ChartQA**。ChartQA是一个专门用于图表问答的大型基准数据集。它旨在评估模型在理解和推理图表内容方面的能力。数据集包含由人类专家制作的图表以及针对这些图表提出的自然语言问题。问题不仅需要模型从图表中提取显式信息，还要求进行逻辑推理和算术运算。

实验选择了ChartQA的测试集部分（包含2500个图像及其问答对）来进行评测。

#### 综合基准测试数据集

当模型的多模态理解和推理能力提升后，CQA的任务表现也会随之提升，所以为了全面评估MLLM的图文多模态能力，我又找了两个极具代表性的基准测试数据集：MME和MMBench。

**MME**是一个全面的多模态评测基准，旨在评估多模态模型的感知和认知能力。它涵盖了14个子任务：物体存在性、计数、位置识别、颜色识别、OCR、常识推理等。能够更加全面地衡量模型的综合能力。

**MMBench**是由[OpenCompass](https://github.com/open-compass/MMBench)社区推出的另一个权威多模态评测基准。它通过精心设计的单选题来评估模型在20个不同能力维度上的表现。这个Benchmark采用了"循环评估"策略，能有效避免数据泄露，更准确地反映模型的水平。

#### 数据集整理

所以，一共要测试**4个数据集**：

| 数据集名称 | 类型 | 描述 |
|------------|------|------|
| 自建的CQA数据集 | I+Q+A | 自建测试集，有简答题 |
| ChartQA数据集 | I+Q+A | 经典的图表问答数据集，单词回答模式 |
| MME基准测试数据集 | I+Q+A | 感知和认知评测，判断题 |
| MMBench基准测试数据集 | I+Q+A | 20维度能力评测，选择题 |

---

## 模型选择

在模型选择上，参照各大榜单和开源数据，选取了两个代表性的开源模型：**InternVL-8B** 和 **Ovis2-8B**。

#### InternVL3-8B

InternVL-8B由上海AI实验室发布，是InternVL3系列的中等规模版本，覆盖文本、图像、视频等多模态任务。


架构上，视觉编码器采用 `InternViT-300M-448px-V2_5`，支持动态分辨率策略（图像分块为448×448像素）和可变视觉位置编码（V2PE），提升了大图的理解能力。语言模型*基于 `InternLM3-8B`通过原生多模态预训练实现视觉与语言的自然对齐，文本性能超越同尺寸的Qwen2.5-7B

#### Ovis2-8B

阿里开源，Ovis2-8B是Ovis2系列的中等规模版本，主打跨模态动态对齐和视频理解能力，支持文本、图像、视频、语音四模态处理。

![Ovis2架构图](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507291616227.png)

Ovis通过引入**视觉嵌入表**实现视觉与文本嵌入策略的结构对齐。传统MLLM中，视觉嵌入由视觉编码器直接生成连续向量，而文本嵌入通过查找表（look-up table）索引离散词向量。
而Ovis简单来说，就是借鉴LLM的索引思想，通过**视觉概率化分词**——将图像块通过线性投影和Softmax生成概率分布，表示其与视觉词汇表中各"视觉词"的关联强度，并基于概率分布从视觉嵌入表中检索多个嵌入向量，加权求和得到最终视觉嵌入。

从严格意义上来讲，这两个模型都是模态融合模型，桥接部分都使用了LLaVA的MLP方案，属于综述调研中的**端到端**模型，而不是两阶段的模型。

---

## 模型部署

部署平台上，使用FunHPC弹性云，配置一张V100显卡。

![部署环境](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507282125888.png)


模型部署上，IntenrnVL3使用**LMDeploy**部署框架，LMDeploy是上海人工智能实验室开发的一款大语言漠型部署工具包，旨在提供高效的推理、量化和服务化功能。其核心组件TurboMind推理引擎使用了持续批处理、优化的KV缓存管理等技术。此外，LMDeploy支持W4A16等多种量化方案，能有效降低模型显存占用。

由于ovis2是由阿里巴巴最新开源的MLLM，暂时没有LMDeploy部署方案，所以使用了Huggingface原生的框架部署。最近发现了Huggingface开始支持Flash-Attention了，可以加速评测速度，但是新版本的Transformers库不支持V100这种旧显卡，所以暂未启用Flash-attn。

[LMDeploy支持的模型](https://lmdeploy.readthedocs.io/zh-cn/latest/supported_models/supported_models.html)

---

## 评测方案

模型部署后，分别测试自建数据集、chartQA数据集、MME、MMBench四个数据集的表现。

自建数据集使用了准确度这个评测指标。针对模型的输出内容，我们将其与标准答案进行对比，若模型输出的关键词全部在标准答案里，就认为模型回答正确。当模型多输出或少输出关键词的时候，认为模型回答错误。

ChartOA是单词问答，同样采用准确率这个评估指标，只需要模型输出一个单词或数字，这样很容易与标准答案进行比较。

MME和MMBench两个基准测试比较成熟，有现有的评测框架的支持，实验中采用了**VLMEvalKit**这个评测框架。在配置好评测文件后可以直接开始测试。

对应的评测流程和代码仓库放在了仓库对应的test目录。

---

## 评测结果

两个模型在四个数据集上的评测对比：

![评测结果总览](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507291502625.png)

自建数据集预测错误样本分布：

![错误样本分布](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507291510997.png)

详细结果存储在了仓库如下文件夹里：

![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507292016963.png)


---

## 评测结果分析

#### 1. CQA任务表现

**自建数据集**

老师给定的数据集总体上有难度，包括很多简答格式的题目。模型需要具备细粒度的特征提取能力，才能够理解图像的层级结构关系(这同样是CQA任务需要攻克的难点)。

事实上，在一些频繁出错的样本(比如ID为27、349、258)上，最先进的闭源模型也会犯错误，比如 Gemini2.5pro 模型在ID为349的图片上同样分辨不清楚“经营管理层的部门组成”。

![错误案例](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507291531885.png)

所以CQA仍然是多模态理解和推理中一个极具挑战的任务。现有的模型在复杂的层级结构处理 、细粒度文本识别、空间关系推理等方面仍然存在短板：

**ChartQA数据集**

InternVL3-14B-4bit 表现最佳(77.36%)，InternVL3-8B 表现次之(75.68%)，Ovis2-8B 表现最差(63.68%)

这两个系列的模型语言基座都是Qwen模型，区别在于InternVL3采用InternViT-300M-448px视觉编码器，支持动态分辨率策略和可变视觉位置编码(V2PE)，这对于理解图表中的精细结构和空间关系非常重要，而相较于Ovis2型，AIMv2视觉编码器就显得很一般了。

另外发现，模型参数规模的扩大对性能指标的提升效果有限。InternVL3-14B版本相较于8B版本仅实现了1.7个百分点的性能提升。可能是由于采用了4bit量化处理，精度有所损失。

>之所以进行模型量化，是因为V100跑不了14B版本，就提前对模型进行了量化处理

#### 2. 通用多模态任务表现

**MME基准测试**

- **感知能力(Perception)**：InternVL3-8B (1730.59) > Ovis2-8B (1597.05)  
- **推理能力(Reasoning)**：Ovis2-8B (685.00) > InternVL3-8B (664.29)

InternVL3在视觉感知任务上表现更强，这得益于其优化的视觉编码器和动态分辨率处理；而Ovis2的MoE架构在复杂推理任务中展现出了更多优势。

总体上InternVL3还是更适合图标理解任务，因为图像的分辨率对表格的识别至关重要，而InternVL3很好的支持了这一点，InternVL3还有更多的架构优势，原论文：https://arxiv.org/pdf/2504.10479


**MMBench测试**
- **开发集**：InternVL3-8B (90.91%) > Ovis2-8B (87.27%)
- **测试集**：Ovis2-8B (84.26%) > InternVL3-8B (82.41%)

这个数据是一个Overall的概括（多项指标的平均数），结果上两个模型在MMBench上的差距并不显著。InternVL3在开发集上表现更优，而Ovis2在测试集上反超，可能是Ovis模型在某些指标上更加突出。详细测评结果在仓库对应的result目录下。

## 参考资料

<details>
<summary>论文合集</summary>

[ChartQA](https://arxiv.org/abs/2203.10244)  
[InternVL3](https://arxiv.org/abs/2504.10479)  
[Ovis](https://arxiv.org/abs/2405.20797)  
[MLLM survey](https://arxiv.org/abs/2306.13549)  
[ChartGemma](https://arxiv.org/abs/2407.04172)  
[CogAgent](https://arxiv.org/abs/2312.08914)  
[MME survey](https://arxiv.org/abs/2411.15296)  

</details>

<details>
<summary>一些链接</summary>

Benchmark:  
https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Benchmarks

Benchmark introduce:   
https://zhuanlan.zhihu.com/p/716280117



Dataset introduce:  
https://zhuanlan.zhihu.com/p/701404377

OpenCompass:  
https://github.com/open-compass/OpenCompass
https://rank.opencompass.org.cn/leaderboard-multimodal
https://muxue.blog.csdn.net/article/details/147155744
https://blog.csdn.net/qq_36803941/article/details/135938692

VLMEvalKit:  
https://github.com/open-compass/VLMEvalKit
https://blog.csdn.net/qq_29788741/article/details/135708616
https://aicarrier.feishu.cn/wiki/TljbwJgjdi7MG1k7a80cWITpnHd

InternVL tutorials  
https://internvl.readthedocs.io/en/latest/index.html#

MMBench:  
https://github.com/open-compass/mmbench/

W8A8 lite:  
https://lmdeploy.readthedocs.io/zh-cn/latest/quantization/w8a8.html

L2G3-LMDeploy:  
https://www.cnblogs.com/zhengzirui/p/18744726?utm_source=chatgpt.com

![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507292042565.png)

</details>



