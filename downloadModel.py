import os
from huggingface_hub import snapshot_download

# 1. 设置镜像站环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 定义下载路径
local_dir = "./models/bert-base-chinese"

print(f"开始从镜像站下载模型到: {local_dir} ...")

try:
    snapshot_download(
        repo_id="google-bert/bert-base-chinese",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.safetensors"] # 只下载 bin 格式
    )
    print("✅ 下载成功！现在你可以重新运行训练脚本了。")
except Exception as e:
    print(f"❌ 下载失败: {e}")