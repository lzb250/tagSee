# config/model_config.py
import os
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 本地模型路径配置
MODEL_CONFIGS = {
    'bert-base-chinese': {
        'local_path': PROJECT_ROOT / 'models' / 'bert-base-chinese',
        'remote_name': 'bert-base-chinese'
    },
    'custom-skill-model': {
        'local_path': PROJECT_ROOT / 'models' / 'skill_extraction_model_final',
        'remote_name': None
    }
}

def get_model_path(model_name: str) -> str:
    """获取模型路径，优先使用本地模型"""
    if model_name in MODEL_CONFIGS:
        local_path = MODEL_CONFIGS[model_name]['local_path']
        if local_path.exists():
            return str(local_path)
        else:
            # 如果本地不存在，返回远程名称（会尝试在线下载）
            return MODEL_CONFIGS[model_name]['remote_name'] or model_name
    return model_name