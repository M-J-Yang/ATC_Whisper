# 快速开始指南

> **v1.1 更新**: 增加数据增强、灵活的编码器控制、ATC词表约束、完整的服务器部署指南

## 📦 完整项目文件清单

已为您创建以下文件：

### 配置文件
- ✅ **config.yaml** - 全局配置（可根据需要调整）
- ✅ **requirements.txt** - Python依赖列表

### 核心脚本
- ✅ **preprocess.py** - 数据预处理管道（包含数据增强：速度扰动、SpecAugment）
- ✅ **train.py** - Whisper模型微调（支持DDP、灵活的编码器控制）
- ✅ **inference.py** - 推理和评估脚本（支持ATC词表约束）
- ✅ **atc_decoder.py** - ATC词汇约束解码器（提高精度）

### 文档
- ✅ **README.md** - 详细项目文档（含服务器部署指南）
- ✅ **QUICKSTART.md** - 本文件

---

## 🚀 执行步骤

### 方式 A：一键执行（推荐）

```bash
cd "d:\NPU_works\语音\demo"
bash run.sh
```

### 方式 B：分步执行

#### 第1步：安装依赖
```bash
pip install -r requirements.txt
```

#### 第2步：数据预处理（~5-10分钟）
```bash
python preprocess.py
```

**检查输出**:
```bash
dir outputs\processed_data\train
dir outputs\processed_data\val
dir outputs\processed_data\test
```

#### 第3步：模型训练（~2-4小时，取决于GPU）

**基础训练** (冻结编码器，快速):
```bash
python train.py
```

**提高精度** (解冻编码器最后4层):
```bash
python train.py --unfreeze-encoder-layers 4
```

**全量微调** (最高精度，需要更多显存):
```bash
python train.py --unfreeze-encoder-layers -1
```

**参数高效** (使用Adapter层):
```bash
python train.py --use-adapter true
```

**监控训练**:
```bash
tensorboard --logdir=logs/
```

然后打开浏览器访问: http://localhost:6006

#### 第4步：推理和评估（~20-30分钟）

**评估整个测试集**:
```bash
python inference.py \
    --model_path outputs/models/final_model \
    --dataset_dir outputs/processed_data \
    --split test \
    --output_dir outputs/results
```

**带ATC词表约束的评估** (提高域特定精度):
```bash
python inference.py \
    --model_path outputs/models/final_model \
    --dataset_dir outputs/processed_data \
    --split test \
    --output_dir outputs/results_constrained \
    --vocab_constraint "ATCOSIM/TXTdata/wordlist.txt"
```

**转录单个文件**:
```bash
python inference.py \
    --model_path outputs/models/final_model \
    --audio_path "path/to/your/audio.wav"
```

**建立Baseline对比** (使用预训练模型):
```bash
python inference.py \
    --model_path "openai/whisper-base" \
    --dataset_dir outputs/processed_data \
    --split test \
    --output_dir outputs/baseline
```

---

## 📊 预期结果

### 数据统计
```
train: ~8,062 条样本
val:   ~1,008 条样本
test:  ~1,008 条样本
总计:  ~10,078 条样本（全部ATCOSIM）
```

### 性能指标（预期）
| 指标 | 预期值 |
|------|--------|
| WER (词错率) | ~45% |
| CER (字错率) | ~25% |
| 推理速度 | 3-4x RT |
| 内存占用 | ~8GB per GPU |

---

## 🔧 常用命令速查

### 数据预处理
```bash
python preprocess.py
```

### 训练选项
```bash
# 基础微调（冻结编码器）
python train.py

# 解冻编码器最后N层
python train.py --unfreeze-encoder-layers 4

# 全量微调
python train.py --unfreeze-encoder-layers -1

# 使用Adapter层（参数高效）
python train.py --use-adapter true

# 自定义配置路径
python train.py --config custom_config.yaml
```

### 推理和评估
```bash
# 评估整个数据集
python inference.py \
    --model_path outputs/models/final_model \
    --dataset_dir outputs/processed_data \
    --split test \
    --output_dir outputs/results

# 带词表约束
python inference.py \
    --model_path outputs/models/final_model \
    --dataset_dir outputs/processed_data \
    --split test \
    --vocab_constraint "ATCOSIM/TXTdata/wordlist.txt"

# 转录单个文件
python inference.py \
    --model_path outputs/models/final_model \
    --audio_path "your_audio.wav"

# Baseline评估
python inference.py \
    --model_path "openai/whisper-base" \
    --dataset_dir outputs/processed_data \
    --split test \
    --output_dir outputs/baseline
```

### 测试ATC词汇约束
```bash
python atc_decoder.py
```

---

## 🎯 优化建议

### 如果精度不足（WER > 50%）
1. **启用数据增强** (推荐首选)
   ```yaml
   data:
     augmentation:
       enabled: true
       speed_perturb: [0.9, 1.1]  # ±10% 速度扰动
   ```

2. **解冻编码器层**
   ```bash
   python train.py --unfreeze-encoder-layers 4
   ```

3. **增加训练时间**
   ```yaml
   training:
     epochs: 15  # 从10增至15
   ```

4. **使用更大的模型**
   ```yaml
   model:
     whisper_size: "small"  # 从base改为small
   ```

5. **应用词表约束** (后处理，零成本)
   ```bash
   python inference.py ... --vocab_constraint "wordlist.txt"
   ```

### 如果推理太慢（RT > 5x）
1. **使用更小的模型**
   ```yaml
   model:
     whisper_size: "tiny"  # 或 "base"
   ```

2. **减小Beam宽度**
   ```yaml
   inference:
     beam_size: 3  # 从5改为3
   ```

3. **禁用Beam Search**
   修改 train.py 中的推理参数：
   ```python
   num_beams=1  # 改为贪心解码
   ```

### 如果显存不足（CUDA OOM）
1. **冻结编码器** (默认配置)
   ```bash
   python train.py  # 仅微调解码器，显存需求最低
   ```

2. **降低batch size**
   ```yaml
   training:
     batch_size: 2  # 从4降至2
   ```

3. **增加梯度累积**
   ```yaml
   training:
     gradient_accumulation_steps: 8  # 保持有效batch_size
   ```

4. **使用更小的模型**
   ```yaml
   model:
     whisper_size: "tiny"  # 或 "base"
   ```

---

## 📁 输出文件位置

```
outputs/
├── processed_data/           ← 预处理的数据
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   └── final_model/          ← 训练好的模型（用于推理）
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
├── logs/
│   └── events.out.*          ← TensorBoard日志
└── results/
    ├── evaluation_report.json ← 评估结果
    └── transcription_results.csv ← 详细转录结果
```

---

## 🔍 故障排除

### 问题：Python模块找不到
```bash
# 确保已安装所有依赖
pip install -r requirements.txt --upgrade

# 检查安装
python -c "import torch; print(torch.__version__)"
```

### 问题：CUDA相关错误
```bash
# 检查GPU
python -c "import torch; print(torch.cuda.is_available())"

# 检查显存
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

### 问题：数据路径错误
```bash
# 检查config.yaml中的路径
python -c "from pathlib import Path; print(Path('d:\\NPU_works\\语音\\demo\\ATCOSIM').exists())"
```

### 问题：模型加载失败
```bash
# 清除缓存并重新下载
rm -rf ~/.cache/huggingface/

# 重新运行脚本会自动下载
python preprocess.py
```

---

## 💾 节省空间的建议

### 清除中间检查点（保留最佳模型）
```bash
# 自动保存的最新3个检查点已由配置管理
# 完成训练后可手动删除
rm -rf outputs/models/checkpoint-*
```

### 压缩输出结果
```bash
# 只需保留以下文件供后续使用
# - outputs/models/final_model/ （必需）
# - outputs/results/evaluation_report.json （可选）
```

---

## 📞 常见问题

**Q: 训练需要多长时间？**
A: 2-4小时（取决于GPU和配置）。使用2张4090时约2小时。

**Q: 可以在单GPU上运行吗？**
A: 可以，将 `config.yaml` 中的 `device_ids: [0]`。

**Q: 支持中文吗？**
A: ATCOSIM是英文数据集。Whisper支持99+语言，可用于中文ASR。

**Q: 模型可以离线使用吗？**
A: 可以。训练完成后，整个模型在 `outputs/models/final_model/` 目录，支持离线推理。

**Q: 如何部署到生产环境？**
A: 参考 `inference.py` 中的 `WhisperInference` 类，可集成到任何Python应用或使用FastAPI创建API服务。

---

## ✅ 验证安装成功

```bash
# 1. 检查GPU
python -c "import torch; assert torch.cuda.is_available()"

# 2. 检查依赖
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration"

# 3. 检查数据集
python -c "from pathlib import Path; assert Path('d:\\NPU_works\\语音\\demo\\ATCOSIM\\TXTdata\\fulldata.csv').exists()"

# 4. 运行演示
python atc_decoder.py
```

所有检查都通过后，您可以开始训练！

---

## 🎓 后续学习

- **了解Whisper**: https://github.com/openai/whisper
- **HuggingFace文档**: https://huggingface.co/docs/transformers
- **ATCOSIM论文**: https://www.uni-sb.de/research/projects/atcosim
- **语音识别基础**: https://distill.pub/2017/ctc/

---

**准备好开始了吗？** 运行 `python preprocess.py` 来处理您的数据！ 🚀
