# CQA-survey


## 综述文章调研：

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

--- 

## 模型选择：

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

---

## 总体流程

### 数据选择
CQA目前开放的数据集比较少，具有代表性的是ChartQA。

ChartQA是一个专门用于图表问答的大型基准数据集。它旨在评估模型在理解和推理图表内容方面的能力。数据集包含由人类专家制作的图表以及针对这些图表提出的自然语言问题。问题不仅需要模型从图表中提取显式信息，还要求进行逻辑推理和算术运算。所以我选择了ChartQA的测试集部分（包含2500个图像及其问答对）来进行评测。

同时，为了全面评估MLLM的图文多模态能力，我又找了两个极具代表性的基准测试数据集MME和MMBench。
MME是一个全面的多模态评测基准，旨在评估多模态模型的感知和认知能力。它涵盖了14个子任务，如物体存在性、计数、位置识别、颜色识别、OCR、常识推理等，能够全面地衡量模型的综合能力。

MMBench是由[OpenCompass](https://github.com/open-compass/MMBench)社区推出的另一个权威多模态评测基准。它通过精心设计的单选题来评估模型在20个不同能力维度上的表现。MMBench的特点是其高质量的题目和“循环评估”策略，能有效避免数据泄露，从而更准确地反映模型的真实水平。

所以，一共测试4个数据集：
- 老师给定的CQA数据集
- ChartQA标准数据集
- MME基准测试数据集
- MMBench基准测试数据集

### 模型选择
在模型选择上，参照各大榜单和开源数据，选取了两个代表性的开源模型InternVL-8B、Ovis2-8B。

InternVL-8B由上海人工智能实验室开源，是InternVL3系列​​ 的中等规模版本，覆盖文本、图像、视频等多模态任务。架构上，视觉编码器​​采用 ​​InternViT-300M-448px-V2_5​​，支持动态分辨率策略（图像分块为448×448像素）和可变视觉位置编码（V2PE），提升长上下文理解能力。​语言模型​​基于 ​​InternLM3-8B-Instruct​​，通过原生多模态预训练实现视觉与语言的自然对齐，文本性能超越同尺寸的Qwen2.5-7B。

Ovis2-8B​由阿里巴巴国际化团队推出，是 ​​Ovis2 系列​​ 的中等规模版本（8B参数），主打跨模态动态对齐和视频理解能力，支持文本、图像、视频、语音四模态处理。
架构设计上，使用了​​改进的注意力机制（含残差连接），构建768维统一嵌入空间，实现文本与视觉的高效交互。
​​混合专家（MoE）架构​​：动态路由系统分配任务至不同专家模块，显存占用降低67%的同时保持高精度。
​​视频处理优化​​：3D卷积+Transformer结合，支持关键帧选择算法（基于DPP和MDP），提升长视频理解效率。

所以从严格意义上来讲，这两个模型都是模态融合模型，桥接部分都使用了LLaVA的MLP方案，属于综述调研中的**端到端**模型，而不是两阶段的模型。

### 模型部署
部署平台上，使用FunHPC弹性云部署，使用V100显卡

![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507282125888.png)

模型部署上，IntenrnVL3使用LMDeploy部署框架，LMDeploy是上海人工智能实验室开发的一款大语言模型部署工具包，旨在提供高效的推理、量化和服务化功能。其核心组件TurboMind推理引擎使用了持续批处理、优化的KV缓存管理等技术。此外，LMDeploy支持W4A16等多种量化方案，能有效降低模型显存占用。

Ovis2是由阿里巴巴最新开源的MLLM，暂时没有LMDeploy部署方案（[LMDeploy支持的模型](https://lmdeploy.readthedocs.io/zh-cn/latest/supported_models/supported_models.html)）,所以使用了Huggingface原生的框架部署。最近发现了Huggingface开始支持Flash-Attention了，可以加速评测速度，但是新版本的Transformers库不支持V100这种旧显卡，所以暂未启用Flash-attn。

### 评测方案

在两个模型部署后，分别测试了自建数据集（老师给定的）、ChartQA数据集、MME、MMBench四个数据集的表现。

自建数据集使用了准确度这个评测指标。针对模型的输出内容，我们将其与标准答案进行对比，若模型输出的关键词全部在标准答案里，就认为模型回答正确。当模型多输出或少输出关键词的时候，认为模型回答错误。

ChartQA是单词问答，同样采用准确率这个评估指标，只需要模型输出一个单词或数字，这样很容易与标准答案进行比较。

MME和MMBench两个基准测试比较成熟，有现有的评测框架的支持，实验中采用了VLMEvalKit这个评测框架。在配置好评测文件后可以直接开始测试。

### 评测结果

测试数据集分为自定义数据集测试和ChartQA标准数据集
| 数据集             | 模型版本           | 准确率      | 模型简介                           | 备注         |
|------------------|-------------------|------------------------|------------------------------------|--------------|
| test_data（自定义） | InternVL3-8B       | 0.714                  | 8B参数量           |              |
| test_data（自定义） | InternVL3-14B-4bit | 0.690                  | 14B参数量，4bit量化 | xx     |
| ChartQA（标准）    | InternVL3-8B       | 0.7568 (1892/2500)     | 8B参数量           |              |
| ChartQA（标准）    | InternVL3-14B-4bit | 0.7736 (1934/2500)     | 14B参数量，4bit量化  | xx     |

详细测试结果存放在文件夹：
./TEST_result
./ChartQA_test

## 参考资料
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


| 数据集类型   | 说明                                   | 适用场景               |
|--------------|----------------------------------------|------------------------|
| MMBench  |         原数据集                               |    包含了训练集额、验证集、测试集                    |
| DEV          | 开发集，用于模型调试和验证             | 开发调试阶段           |
| TEST         | 测试集，用于最终性能评估               | 正式评测               |
| V11          | 升级版本，修复了原版的一些问题         | 推荐使用               |
| MINI         | 精简版，样本数量大幅减少               | 快速测试               |
| CN/EN        | 中文/英文版本                          | 语言特定评测           |


