# 使用方法

## 使用的模型及数据
qwen2.5-0.5b: \
https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct \
siglip: \
https://hf-mirror.com/google/siglip-base-patch16-224

### 数据集
1、预训练数据：\
图片数据：\
https://hf-mirror.com/datasets/liuhaotian/LLaV  A-CC3M-Pretrain-595K \
中文文本数据：\
https://hf-mirror.com/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions \
2、SFT数据:\
图片数据:\
https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset \
中文文本数据:\
https://hf-mirror.com/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions

## 训练
预训练:\
python train.py\
SFT:\
python sft_train.py

## 测试
python test.py
