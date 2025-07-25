# from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
# from lmdeploy.vl import load_image
# model = 'OpenGVLab/InternVL3-8B'
# image = load_image('/data/coding/test_data/image/27.jpg')
# pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=16384, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
# response = pipe(('董事会包含哪些部门？', image))
# print(response.text)

# 启用以下命令行工具运行模型服务
# lmdeploy serve api_server \
#     /data/coding/InternVL3-8B \
#     --quant-policy 0 \
#     --cache-max-entry-count 0.2\
#     --server-name 0.0.0.0 \
#     --server-port 23333 \
#     --tp 1

# 启用以下命令行工具进行模型量化
# lmdeploy lite auto_awq \
#    /data/coding/InternVL3-14B \
#   --calib-dataset 'ptb' \
#   --calib-samples 128 \
#   --calib-seqlen 2048 \
#   --w-bits 4 \
#   --w-group-size 128 \
#   --batch-size 1 \
#   --search-scale False \
#   --work-dir /data/coding/InternVL3-14B-4bit

