# 使用LMDeploy部署与批量推理

[官方文档](https://lmdeploy.readthedocs.io/zh-cn/latest/index.html)

## 1. LMDeploy
LMDeploy是由OpenMMLab团队开发的大模型部署工具包，专门为大规模语言模型和多模态模型的高效部署而设计。

## 2. LMDeploy服务部署

### 2.1 环境准备
```bash
# 安装LMDeploy
pip install lmdeploy
```

### 2.2 模型部署

基础部署
```bash
# 部署InternVL3模型
lmdeploy serve api_server \
    /path/to/internvl3-model \
    --backend turbomind \
    --server-port 23333
```
带参数的部署
```bash
# 带更多配置的部署命令
lmdeploy serve api_server \
    /path/to/internvl3-model \
    --backend turbomind \
    --server-port 23333 \
    --server-name 0.0.0.0 \
    --instance-num 1 \
    --tp 1 \
    --session-len 32768 \
    --cache-max-entry-count 0.8
```

各个参数设置我参考了[官方文档](https://lmdeploy.readthedocs.io/zh-cn/latest/multi_modal/vl_pipeline.html)

参数说明
- `--backend`: 推理后端，可选turbomind或pytorch
- `--server-port`: 服务端口号，我使用了默认值
- `--server-name`: 服务地址，0.0.0.0表示监听所有网络接口，同样是默认值
- `--instance-num`: 实例数量，1即可
- `--tp`: Tensor并行度，1即可
- `--session-len`: 最大会话长度，300足够了
- `--cache-max-entry-count`: 缓存占用显存比例,因为我的是一块老的V100-32GB，这里设置的不能太大，比如InternVL3-8B使用fp16精度的话，占用显存就是8*2 = 16GB, 另外需要8 *0.5 = 4GB存储传播梯度和优化器参数。最后还剩下32-16-8 = 8GB，所以我设置了这个参数为0.2,代表我只0.2倍的剩余显存。

### 2.3 验证部署
部署成功后，可以通过以下方式验证：
```bash
curl http://localhost:23333/v1/models
```

## 3. 评测流程详解

本项目提供了两种评测方式：基于LMDeploy API和直接使用HuggingFace模型。

### 3.1 数据集格式

评测使用的问答数据集格式如下：

每个问答对包含：
- 图片ID（对应图片文件名，不含扩展名）
- 问题（以Q：开头）
- 答案（以A：开头）

我们就可以利用这个格式的特性提取出问答对，做批量推理。

### 3.2 基于LMDeploy API的评测 (TEST_test_lmdeploy.py)

直接参考openai接口格式就行，是兼容的，但这里注意将图片转成了base64编码，用于适配lmdeploy接口

**图片Base64编码**
```python
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
```

**问答文件解析**
```python
def parse_qa_file(qa_file_path):
    # 读取并解析Q&A.txt文件
    # 返回包含image_id, question, expected_answer的字典列表
```

#### 评测流程

1. **数据预处理**：解析问答文件，提取图片ID、问题和预期答案
2. **图片加载**：根据图片ID加载对应的图片文件
3. **API调用**：将图片转换为base64格式，构建API请求
4. **结果收集**：收集模型回答并与预期答案对比
5. **结果保存**：将评测结果保存到文件




#### **项目结构：**
```
TEST_test_my_own_data/
├── README.md                  # 本文档
├── TEST_test_lmdeploy.py     # LMDeploy API评测脚本
└── TEST_test_hf.py           # HuggingFace评测脚本
```

