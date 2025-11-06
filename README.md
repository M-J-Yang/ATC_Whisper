# ATC 语音识别系统

基于 Whisper 的航空交通管制 (ATC) 语音识别系统，支持命令行和 Web 界面两种使用方式。

## 📋 项目概述

- **数据集**: ATCOSIM (10小时，10078条话语)
- **模型**: OpenAI Whisper-base
- **使用方式**: 命令行 + Web 界面
- **目标**: 平衡精度(WER ~45%)和推理速度(RTF ~0.12)

## 📁 项目结构

```
demo/
├── core/                      # 核心模块
│   ├── inference.py          # Whisper 推理引擎
│   ├── preprocess.py         # 数据预处理
│   ├── train.py              # 模型训练
│   └── atc_decoder.py        # ATC 词汇约束解码器
├── backend/                  # Web 后端
│   ├── app.py               # FastAPI 应用
│   └── inference_service.py  # 推理服务
├── frontend/                 # Web 前端
│   └── src/components/      # React 组件
├── scripts/                  # 命令行工具
│   ├── inference_single.py  # 单条推理
│   └── inference_interactive.py # 交互式推理
├── models/                   # 模型文件
├── config.yaml              # 配置文件
└── start_all.ps1            # 一键启动
```

详细结构见 [FILE_STRUCTURE.md](FILE_STRUCTURE.md) 或 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 🚀 快速开始

### 方式一：Web 应用（推荐）

#### 1. 安装依赖
```powershell
.\install_dependencies.ps1
```

#### 2. 一键启动
```powershell
.\start_all.ps1
```

系统会自动启动后端 (http://localhost:8000) 和前端 (http://localhost:3000)

#### 3. 使用界面
1. 点击"加载模型"
2. 选择推理方式：单条推理/实时识别
3. 查看结果并导出

详见 [README_WEB.md](README_WEB.md)

### 方式二：命令行工具

#### 单次推理
```bash
python scripts/inference_single.py
```

#### 交互式推理
```bash
python scripts/inference_interactive.py
```

### 方式三：训练模型

#### 1️⃣ 环境安装

```bash
# Python 3.10+
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

#### 2️⃣ 数据预处理

```bash
python core/preprocess.py
```

**流程**:
- 加载 `fulldata.csv` 元数据
- 加载并重采样音频到16kHz
- 标准化转录文本 (移除 ~p ~s ~a 特殊标记)
- 按说话人分层划分数据集 (train 80% / val 10% / test 10%)
- 保存标准化的JSON清单

**输出**:
```
outputs/processed_data/
├── train/
│   ├── *.wav (重采样的音频)
│   └── train_manifest.json
├── val/
│   ├── *.wav
│   └── val_manifest.json
└── test/
    ├── *.wav
    └── test_manifest.json
```

#### 3️⃣ 模型训练

```bash
# 基础训练
python core/train.py

# 解冻最后N层编码器
python core/train.py --unfreeze-encoder-layers 2

# 使用Adapter层
python core/train.py --use-adapter true
```

**特点**:
- ✅ 自动检测GPU，启用DDP分布式训练
- ✅ 灵活的编码器冻结策略（冻结/部分/全量微调）
- ✅ 支持Adapter层参数高效微调
- ✅ FP32精度训练（Windows兼容，可通过配置启用FP16）
- ✅ 评估步长可配置
- ✅ 自动保存最好的模型

**训练配置** (来自 `config.yaml`):
- Batch Size: 4 (可根据显存调整)
- Learning Rate: 1e-5
- Epochs: 10
- Warmup Steps: 500
- Mixed Precision: FP32 (稳定性优先)

**输出**:
```
outputs/
├── models/final_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── preprocessor_config.json
│   └── tokenizer_vocab.json
├── logs/
│   └── tensorboard events
└── checkpoints/
    └── checkpoint-*/ (中间检查点)
```

#### 4️⃣ 推理和评估

使用核心推理引擎：
```bash
# 单个文件推理
python core/inference.py \
    --model_path models/final_model \
    --audio_path /path/to/audio.wav

# 整个测试集评估
python core/inference.py \
    --model_path models/final_model \
    --dataset_dir processed_data \
    --split test \
    --output_dir outputs/results
```

或使用命令行工具（更方便）：
```bash
# 单条推理
python scripts/inference_single.py

# 交互式推理
python scripts/inference_interactive.py
```

#### 评估带词表约束（用于对比）
```bash
python inference.py \
    --model_path outputs/models/final_model \
    --dataset_dir outputs/processed_data \
    --split test \
    --output_dir outputs/results_constrained \
    --vocab_constraint "ATCOSIM/TXTdata/wordlist.txt"
```

**输出**:
```
outputs/results/
├── evaluation_report.json (详细指标: WER, CER, 按说话人分析)
└── transcription_results.csv (转录结果)
```

## 📊 配置参数

编辑 `config.yaml` 调整:

```yaml
# 数据配置
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  target_sr: 16000
  normalize: true  # 音量归一化

  # 数据增强配置
  augmentation:
    enabled: true
    speed_perturb: [0.9, 1.1]  # 速度扰动: ±10%
    # SpecAugment参数: freq_mask_param, time_mask_param

# 模型配置
model:
  type: "whisper"
  whisper_size: "base"  # tiny|base|small|medium|large
  use_atc_vocab_constraint: true

# 训练配置
training:
  epochs: 10
  batch_size: 4  # 根据显存调整 (4090推荐4)
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  warmup_steps: 500
  weight_decay: 0.01

  # DDP多GPU配置
  distributed: true
  device_ids: [0]  # 修改为实际GPU数量

  # 评估和保存
  eval_steps: 1000
  save_steps: 1000

# 推理配置
inference:
  beam_size: 5        # 柬寨搜索宽度
  max_length: 224     # 最大生成长度
  temperature: 0.0    # 0=确定性, >0=随机
  language: "en"      # en|zh|等

# 系统配置
system:
  seed: 42
  num_workers: 4      # 数据加载进程数
  max_grad_norm: 1.0
  mixed_precision: "fp32"  # fp32|fp16|bf16
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `batch_size` | 每张GPU的批次大小 | 4 (RTX4090), 2 (RTX3090) |
| `grad_accumulation_steps` | 梯度累积步数 | 4 (等效batch=16) |
| `learning_rate` | 学习率 | 1e-5 (保守) ~ 5e-5 (激进) |
| `speed_perturb` | 速度扰动范围 | [0.9, 1.1] (±10%) |
| `beam_size` | 束搜索宽度 | 5 (平衡), 3 (快速), 10 (精度) |
| `max_length` | 最大生成长度 | 224 (Whisper标准) |

## 📈 性能预期

| 阶段 | WER | 推理速度 | 显存占用 |
|------|-----|---------|---------|
| 基础模型(无微调) | ~75% | 3.2x RT | 6GB |
| 微调后(冻结编码) | ~48% | 3.5x RT | 8GB |
| 微调后(全量) | ~42% | 3.5x RT | 16GB |
| +词表约束 | ~38% | 2.8x RT | 8GB |

**说明**:
- RT = Real-Time (处理1小时音频需要的时间)
- 显存占用指单GPU, batch_size=4
- 词表约束是后处理，零额外显存开销

## ✨ 最近更新 (v1.1)

### 新增功能
- ✅ **数据增强**: 实现速度扰动 (±5-10%) 和 SpecAugment
- ✅ **灵活的编码器控制**:
  - `--unfreeze-encoder-layers N` 解冻最后N层
  - `--unfreeze-encoder-layers -1` 全量微调
- ✅ **词表约束**: `--vocab_constraint` 参数用于ATC域约束
- ✅ **服务器部署**: 完整的部署指南和资源需求表
- ✅ **Baseline对比**: 标准化的评估流程和对比框架

### 改进
- 改进text normalization文档（~p=停顿, ~s=不清, ~a=口音）
- 完善了服务器部署流程和日志监控建议
- 添加了FP32/FP16/BF16精度选项说明

## 🏗️ 项目结构

```
demo/
├── config.yaml              # 全局配置
├── requirements.txt         # Python依赖
├── preprocess.py            # 数据预处理脚本
├── train.py                 # 训练脚本 (支持DDP)
├── inference.py             # 推理和评估脚本
├── run.sh                   # 完整流程脚本
├── README.md                # 本文档
├── ATCOSIM/                 # 数据集
│   ├── WAVdata/             # 音频文件
│   ├── TXTdata/
│   │   ├── fulldata.csv     # 元数据
│   │   └── wordlist.txt     # ATC词汇表
│   ├── HTMLdata/
│   └── DOC/
└── outputs/
    ├── processed_data/      # 预处理的数据
    ├── models/              # 训练的模型
    ├── logs/                # TensorBoard日志
    └── results/             # 推理结果
```

## 🔍 关键特性

### 1. 数据预处理
- ✅ 自动移除ATC特殊标记 (~p=停顿, ~s=不清, ~a=口音)
- ✅ 音频重采样和归一化 (16kHz)
- ✅ **数据增强**: 速度扰动 (±5-10%) 和 SpecAugment
- ✅ 分层数据划分 (按说话人)
- ✅ 质量检查 (移除损坏的音频)

### 2. 灵活的训练策略
- ✅ 编码器冻结（快速微调）
- ✅ 部分解冻最后N层（平衡精度和速度）
- ✅ 全量微调（最高精度）
- ✅ DDP多GPU分布式训练
- ✅ 梯度累积支持
- ✅ FP32/FP16/BF16精度选项

### 3. 推理优化
- ✅ 单文件和批量推理
- ✅ **ATC词表约束**（后处理，提高域特定精度）
- ✅ 可调的Beam Search宽度
- ✅ GPU加速推理

### 4. 评估指标
- ✅ 词错率 (WER)
- ✅ 字错率 (CER)
- ✅ 按说话人的WER分析
- ✅ 详细的转录结果导出 (CSV/JSON)

## 🖥️ 服务器部署

### 快速部署流程

```bash
# 1. 克隆/上传项目
cd /path/to/project

# 2. 创建Python虚拟环境
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac

# 3. 安装依赖（确保torch/torchaudio版本一致）
pip install -r requirements.txt

# 4. 验证CUDA/GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# 5. 数据预处理（单次运行）
python preprocess.py

# 6. 启动训练（支持后台运行）
nohup python train.py > train.log 2>&1 &

# 7. 监控训练进度
tail -f train.log
tensorboard --logdir=logs/

# 8. 评估模型
python inference.py \
    --model_path models/final_model \
    --dataset_dir outputs/processed_data \
    --split test \
    --output_dir outputs/results
```

### 关键部署建议

1. **依赖管理**: 保证 `torch==2.x` 和 `torchaudio==2.x` 版本一致
2. **显存管理**: 根据GPU显存调整 `batch_size` (4090 推荐设为4)
3. **数据路径**: 更新 `config.yaml` 中的绝对路径
4. **后台运行**: 使用 `nohup` 或 `tmux/screen` 保持进程
5. **日志监控**: 定期检查 `logs/` 和 `train.log`

### 服务器资源需求

| 配置 | GPU | 显存 | CPU | 内存 |
|------|-----|------|-----|------|
| 最小 | 1x RTX3090 | 24GB | 8核 | 32GB |
| 推荐 | 2x A100 | 80GB | 16核 | 128GB |
| 开发 | 1x RTX4090 | 24GB | 8核 | 64GB |

## 💡 性能优化建议

### 提高精度
1. 增加训练轮数: `training.epochs: 15-20`
2. 使用更大的模型: `model.whisper_size: "small"` or `"medium"`
3. **启用数据增强**: `data.augmentation.enabled: true` (速度扰动+SpecAugment)
4. 解冻编码器层: `python train.py --unfreeze-encoder-layers 4`
5. 调整学习率: `training.learning_rate: 5e-5`

### 加速推理
1. 使用更小的模型: `model.whisper_size: "tiny"` or `"base"`
2. 减小Beam Search宽度: `inference.beam_size: 3`
3. **应用词表约束**: 减少搜索空间，加速解码
4. 量化模型 (需要额外实现)

### 减少显存占用
1. 降低batch size: `training.batch_size: 4`
2. 启用梯度检查: 在train.py中设置
3. 使用更小模型
4. **使用Adapter层**: `python train.py --use-adapter true` (参数减少90%)

## 📊 Baseline对比与评估

### 建立Baseline (微调前)

```bash
# 1. 使用预训练模型(无微调)进行评估
python inference.py \
    --model_path "openai/whisper-base" \
    --dataset_dir outputs/processed_data \
    --split test \
    --output_dir outputs/baseline

# 输出: baseline WER
```

### 评估微调后的模型

```bash
# 2. 使用微调后的模型评估（相同的normalization）
python inference.py \
    --model_path models/final_model \
    --dataset_dir outputs/processed_data \
    --split test \
    --output_dir outputs/finetuned
```

### 对比分析

```bash
# 3. 对比两个评估结果
# 比较 outputs/baseline/evaluation_report.json 和 outputs/finetuned/evaluation_report.json
# 关键指标:
#   - WER降幅: (baseline_wer - finetuned_wer) / baseline_wer
#   - CER降幅
#   - 按说话人分析 (speaker_wers)
```

**示例对比**:
```
| 模式 | WER | CER | 改进 |
|------|-----|-----|------|
| Baseline (无微调) | 75.2% | 32.1% | - |
| 微调后 (10小时) | 48.3% | 18.5% | 36% |
| +词表约束 | 42.1% | 16.2% | 44% |
```

### 确保公平对比

⚠️ **重要**: 确保baseline和微调模型使用**相同的**:
1. ✅ Text normalization (都经过 `preprocess.py` 的normalize_transcription)
2. ✅ 数据划分 (相同的test_manifest.json)
3. ✅ 评估参数 (beam_size, max_length等)
4. ✅ 推理配置 (FP32/FP16, 设备等)

## 🐛 故障排除

### CUDA OOM错误
```python
# 降低batch size in config.yaml
training:
  batch_size: 4  # 从8降至4
  gradient_accumulation_steps: 4  # 增加累积步数
```

### 数据集加载失败
```bash
# 检查数据集路径
ls -la "d:\NPU_works\语音\demo\ATCOSIM"

# 验证fulldata.csv
head -5 "d:\NPU_works\语音\demo\ATCOSIM\TXTdata\fulldata.csv"
```

### 模型加载失败
```bash
# 确保模型目录正确
ls -la outputs/models/final_model/

# 重新下载预训练模型
python -c "from transformers import WhisperForConditionalGeneration; \
    WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')"
```

## 📚 参考资源

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [ATCOSIM Corpus](https://www.uni-sb.de/en/research/projects/atcosim)
- [WER计算方法](https://en.wikipedia.org/wiki/Word_error_rate)

## 📝 许可证

本项目遵循MIT许可证。ATCOSIM数据集有其专属许可证，请查看 `ATCOSIM/DOC/` 目录。

## 👨‍💼 作者

Created with ❤️ for ATC Speech Recognition
