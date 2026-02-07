# 从 CPU 训练切换到 NVIDIA 显卡训练配置指南

## 📋 修改概述

本文档记录了将项目从 CPU 训练切换到 NVIDIA 显卡（CUDA）训练所需的配置和代码修改。

## 🔧 代码修改

### 修改文件：`src/cross_platform_utils.py`

**修改位置**：第 75-84 行

**修改前**：
```python
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
```

**修改后**：
```python
def get_torch_device():
    """获取合适的 PyTorch 设备，自动选择CUDA/MPS/CPU"""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

### 修改说明

- **移除了**训练模式强制使用 CPU 的环境变量检查逻辑
- **移除了** `TRAINING_MODE` 相关代码
- **保留了**自动检测 CUDA/MPS/CPU 的功能
- 使训练和推理都自动使用可用的 GPU 设备

## 📦 环境配置步骤

### 1. 检查 NVIDIA 驱动安装

```bash
# 检查 NVIDIA 驱动是否正确安装
nvidia-smi
```

预期输出应显示：
- NVIDIA 驱动版本
- CUDA 版本
- GPU 型号信息

### 2. 安装 CUDA 版本的 PyTorch

**如果当前安装的是 CPU 版本的 PyTorch**：

```bash
# 卸载当前版本的 PyTorch
pip uninstall torch torchvision torchaudio

# 根据你的 CUDA 版本选择对应的 PyTorch 版本

# 方案一：CUDA 12.1（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 方案二：CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 方案三：CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**注意**：请在远程 Linux 服务器上执行上述命令（MacOS 本身不支持 NVIDIA CUDA）。

### 3. 验证 PyTorch CUDA 支持

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')"
```

预期输出示例：
```
CUDA available: True
Device count: 1
Current device: 0
```

### 4. 验证设备选择功能

```python
# 在 Python 中测试
from src.cross_platform_utils import get_torch_device
print(f"训练将使用的设备: {get_torch_device()}")
```

预期输出：
```
训练将使用的设备: cuda
```

## 🚀 训练命令

### 使用默认参数训练

```bash
python train_offline.py
```

### 使用 GPU 优化的参数训练

由于 GPU 显存通常比系统内存大，可以适当增加 `batch_size`：

```bash
# 使用更大的 batch_size 提高训练效率
python train_offline.py --batch_size 32 --epochs 10

# 或者根据 GPU 显存调整（假设有更多显存）
python train_offline.py --batch_size 64 --epochs 10 --synthetic_samples 5000
```

### 训练参数建议

| GPU 显存 | 推荐 batch_size | 说明 |
|---------|----------------|------|
| 8GB | 16-24 | 适合入门级显卡 |
| 12GB | 24-32 | 适合中端显卡 |
| 16GB+ | 32-64 | 适合高端显卡 |

## 📝 不需要修改的其他配置

### `src/model_training.py`

该文件中的 `TrainingArguments` 配置无需修改，HuggingFace Trainer 会自动：

- 检测可用的 GPU 设备
- 将模型移动到 GPU
- 在每个设备上使用指定的 `per_device_train_batch_size`

现有配置已经是 GPU 友好的：

```python
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size, # 每个 GPU 设备的批次大小
    per_device_eval_batch_size=batch_size,
    # ... 其他参数
)
```

### `src/inference.py`

推理模块已经使用 `get_torch_device()` 函数获取设备，因此会自动使用 GPU：

```python
self.device = get_torch_device()
# 自动选择最佳的可用设备（CUDA/MPS/CPU）
```

## ✅ 验证清单

在开始训练前，请确认以下项目：

- [ ] 已修改 `src/cross_platform_utils.py`，移除了 CPU 强制限制
- [ ] NVIDIA 驱动已正确安装（`nvidia-smi` 可用）
- [ ] 已安装 CUDA 版本的 PyTorch
- [ ] 运行验证命令确认 `torch.cuda.is_available()` 返回 `True`
- [ ] 使用 `get_torch_device()` 测试返回 `cuda`
- [ ] 根据显卡显存调整了合适的 `batch_size`

## ⚠️ 注意事项

### MacOS 本地开发

如果你在 MacOS 本地开发，但使用远程 Linux 服务器进行训练：

- **本地代码**：修改后的代码同时在 Mac 和 Linux 环境下工作
    - MacOS：会使用 MPS（Metal Performance Shaders）或降级到 CPU
    - Linux：会使用 CUDA（如果 NVIDIA 显卡可用）

- **远程训练**：确保在安装了 NVIDIA 驱动和 CUDA 的 Linux 服务器上运行训练脚本

### 显存监控

训练过程中可以使用以下命令监控 GPU 显存使用：

```bash
# 实时监控 GPU 状态
watch -n 1 nvidia-smi
```

### 训练速度对比参考

- **CPU 训练**：~10-20 samples/sec
- **GPU 训练**：~100-300+ samples/sec（取决于显卡型号）

预期速度提升：**10-30 倍**

## 🔍 故障排查

### 问题 1：`torch.cuda.is_available()` 返回 False

**可能原因**：
- 安装了 CPU 版本的 PyTorch
- NVIDIA 驱动未正确安装
- CUDA 版本与 PyTorch 版本不匹配

**解决方案**：
```bash
# 重新安装 CUDA 版本的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 问题 2：训练时显存溢出（OOM）

**解决方案**：
- 减小 `batch_size`（例如从 32 降到 16）
- 减小 `max_length`（如果支持自定义）
- 使用梯度累积（需要在 `TrainingArguments` 中添加 `gradient_accumulation_steps`）

### 问题 3：代码仍使用 CPU 训练

**验证方法**：
```python
# 在训练开始前添加调试信息
import torch
print(f"训练设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**可能原因**：
- 代码修改未生效，需要确认 `src/cross_platform_utils.py` 文件已正确修改
- PyTorch 版本仍为 CPU 版本

## 📞 技术支持

如有问题，请检查：
1. PyTorch 版本：`pip show torch`
2. CUDA 版本：`nvidia-smi`
3. GPU 可用性：`python -c "import torch; print(torch.cuda.is_available())"`

---

**文档版本**：1.0
**最后更新**：2026-02-06
**修改人员**：轶森 (liuzhengbang.lzb)