# src/cross_platform_utils.py
import os
import sys
import platform
from pathlib import Path
from typing import Union, List

def get_platform_info():
    """获取平台信息"""
    return {
        'system': platform.system().lower(),
        'machine': platform.machine().lower(),
        'python_version': platform.python_version()
    }

def setup_encoding():
    """设置正确的编码"""
    if sys.platform == "win32":
        # Windows 控制台编码处理
        try:
            import locale
            locale.setlocale(locale.LC_ALL, '')
        except:
            pass

        # 设置标准输出编码
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        elif hasattr(sys.stdout, 'buffer'):
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 确保环境变量正确
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def safe_path(path: Union[str, Path]) -> Path:
    """安全的路径处理，兼容所有平台"""
    if isinstance(path, str):
        # 处理 Windows 路径分隔符
        path = path.replace('\\', '/').replace('//', '/')
        return Path(path)
    return path

def ensure_directory_exists(directory: Union[str, Path]):
    """确保目录存在"""
    directory = safe_path(directory)
    directory.mkdir(parents=True, exist_ok=True)

def get_torch_device():
    """获取合适的 PyTorch 设备（训练时强制使用 CPU）"""
    import torch
    import os

    # 检查是否在训练模式
    if os.environ.get('TRAINING_MODE', '0') == '1':
        print("🔧 训练模式：强制使用 CPU")
        return torch.device("cpu")

    # 推理时可以使用 GPU/MPS
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def normalize_line_endings(text: str) -> str:
    """标准化换行符"""
    return text.replace('\r\n', '\n').replace('\r', '\n')

def safe_print(*args, **kwargs):
    """安全的打印函数，处理编码问题"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 如果遇到编码错误，尝试用 errors='replace' 处理
        encoded_args = []
        for arg in args:
            if isinstance(arg, str):
                encoded_args.append(arg.encode('utf-8', errors='replace').decode('utf-8'))
            else:
                encoded_args.append(str(arg))
        print(*encoded_args, **kwargs)