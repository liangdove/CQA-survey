## 使用VLMEvalKit评测框架评测MME & MMBench

1. ms下载模型
```
# 以Ovis模型为例子
modelscope download --model AIDC-AI/Ovis2-8B --local_dir /data/coding/Model/Ovis2-8B
```

2. 克隆项目
```
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

注意配置环境的时候tranformers的版本要根据具体模型而配置，比如Ovis2模型明确要求tranformers版本库为4.45.0.
关于Transformers库的历史版本整理如下：
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507281934276.png)

3. 修改模型配置
```
vlmeval/config.py
```
以修改Ovis2-8B模型为例子：  
- 首先修改掉模型路径（model_path参数）
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507281904526.png)

- 然后禁用flash-attn
因为显卡是老机皇V100，不是RTX显卡架构，不能用flash-attn。具体要修改两个地方：
```
vlm/ovis/ovis.py
```
模型定义层中，改掉attention的实现方式，禁用flash-attn
![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507281911314.png)

由于下载到本地的模型未经VLMEvalKit检查，所以手动检查一下其配置文件，手动禁用flash-attn

![](https://cdn.jsdelivr.net/gh/liangdove/PicGo/imgs/202507281914620.png)

如图，将config.json中的"llm_attn_implementation"参数改为eager模式即可。

4. 然后运行评测命令，参数设置参照[官方文档](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/Quickstart.md)即可
```
py运行：
python run.py --data MMBench_V11_MINI --model xxx --verbose

torchrun运行：
torchrun --nproc-per-node=8 run.py --data MMBench_V11_MINI --verbose
```

5. 评测结果：

在一张V100上，一共跑了5个小时，可见如果不做模型切分，评测流程还是挺慢的。如果想要更快可以使用[lmdeploy](https://lmdeploy.readthedocs.io/zh-cn/latest/supported_models/supported_models.html)进行部署，或者看看选用的模型有没有Transformers库的优化支持（一般要求Transformers库版本大于4.50.xx才能做falsh-attn优化）

[MME InternVL3-8B评测结果](../../MME_result/MME_result_InternVL3-8B/InternVL3-8B/InternVL3-8B_MME_score.csv)

[MMBench InternVL3-8B评测结果](../../MMBench_result/MMBench_result_InternVL3-8B/InternVL3-8B/InternVL3-8B_MMBench_V11_MINI_acc.csv)

[MME Ovis2-8B评测结果](../../MME_result/MME_result_Ovis2-8B/Ovis2-8B/Ovis2-8B_MME_score.csv)

[MMBench Ovis2-8B评测结果](../../MMBench_result/MMBench_result_Ovis2-8B/Ovis2-8B/Ovis2-8B_MMBench_V11_MINI_acc.csv)

