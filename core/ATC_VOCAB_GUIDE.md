# ATC语音识别 - 词汇约束完整指南

## 📋 总体方案

不是事后纠正，而是从根本上改进模型，让它在训练时就学会ATC特定词汇，推理时一次性输出正确结果。

## 🚀 快速开始（3步）

### 第1步：构建ATC专用词汇库
```bash
python build_atc_vocab.py
```

**输出文件：**
- `atc_vocab/atc_vocab_classified.json` - 分类词汇表（天气、飞行、部件等）
- `atc_vocab/atc_vocab.txt` - 完整词表
- `atc_vocab/vocab_stats.json` - 词汇统计
- `atc_vocab/atc_vocab_config.yaml` - 训练配置

**示例输出：**
```
天气相关 (45词): 天气, 风, 雨, 云, 能见度, ...
飞行操作 (38词): 跑道, 标高, 转弯, 下降, ...
飞机部件 (32词): 油箱, 发动机, 襟翼, 方向舵, ...
方向度数 (28词): 左, 右, 度, 向, ...
```

### 第2步：审查并补充词汇表（可选）
编辑生成的词汇配置，确保包含所有关键词：
```bash
cat atc_vocab/vocab_stats.json  # 查看统计
```

如需手动补充词汇，编辑：
```bash
atc_vocab/atc_vocab_config.yaml
```

### 第3步：用词汇约束进行微调
```bash
python train_with_vocab_constraint.py
```

**训练参数：**
- 输入数据：`outputs/processed_data/` (需要预处理)
- 输出模型：`outputs/whisper_atc_constrained/final_model`
- 约束权重：1.0（可调整）
- 训练轮数：10轮

**预期结果：**
- CER从26.85% 降低到 ~8-12%
- 特定词汇识别正确率从30% 提升到 90%+

## 📊 验证效果

### 推理时使用约束模型
```bash
# 使用微调后的约束模型推理
python inference.py --model_path outputs/whisper_atc_constrained/final_model \
                    --use_processed --split test
```

### 对比效果
```bash
# 查看改进前后的对比
python evaluate_improvement.py
```

## 🔍 词汇约束的工作原理

### 训练阶段
1. **词汇权重**：ATC词汇的损失权重更高（1.5-1.8倍）
2. **词汇惩罚**：不在ATC词汇表中的词会被惩罚
3. **覆盖率约束**：保证模型能生成95%的约束词汇

### 推理阶段
1. **Beam Search约束**：在decoding时限制候选词为ATC词汇
2. **Log-probability过滤**：过滤置信度低的非约束词
3. **后处理纠正**：如有遗漏，自动纠正

## 📁 文件结构

```
core/
├── build_atc_vocab.py              # 构建ATC词汇库
├── train_with_vocab_constraint.py  # 词汇约束微调
├── inference.py                    # 推理（支持词汇约束）
├── atc_vocab/                      # ATC词汇库（自动生成）
│   ├── atc_vocab_classified.json   # 分类词汇
│   ├── atc_vocab.txt               # 完整词表
│   ├── vocab_stats.json            # 统计信息
│   └── atc_vocab_config.yaml       # 训练配置
└── outputs/
    ├── processed_data/             # 预处理数据
    └── whisper_atc_constrained/    # 微调后的模型
        └── final_model/
            ├── model.safetensors
            ├── processor_config.json
            └── ...
```

## 🎯 具体改进示例

### 问题样本

| 参考 | 旧识别 | CER | 新识别（预期） | CER |
|------|--------|-----|---------------|-----|
| 呃碧空 | 呃避空 | 33% | 呃碧空 | 0% |
| 副油箱 | 投掉复油箱 | 50% | 副油箱 | 0% |
| 浦江 | 普江 | 50% | 浦江 | 0% |
| 稳定 | 稳杆 | 50% | 稳定 | 0% |
| 无限 | 5线 | 80% | 无限 | 0% |

## ⚙️ 高级配置

### 调整词汇约束权重
编辑 `atc_vocab/atc_vocab_config.yaml`：
```yaml
training:
  vocab_constraint_weight: 1.5      # 增加权重提高约束强度
  use_vocab_penalty: true
  vocab_coverage_threshold: 0.95
```

### 按类别调整权重
```yaml
vocabulary_constraints:
  aircraft_parts:
    weight: 2.0   # 飞机部件最重要，权重最高
  flight_operations:
    weight: 1.8
  weather:
    weight: 1.5
```

## 📈 预期改进

| 指标 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| 全局CER | 26.85% | ~10% | 62% ↓ |
| 飞机部件识别 | 30% | 95% | 217% ↑ |
| 天气术语识别 | 45% | 92% | 104% ↑ |
| 推理延迟 | 74s | ~70s | 5% ↓ |

## 🔗 相关命令快速参考

```bash
# 1. 构建词汇库（一次性）
python build_atc_vocab.py

# 2. 查看词汇统计
cat atc_vocab/vocab_stats.json | grep -A 5 "aircraft"

# 3. 微调模型（需要GPU）
python train_with_vocab_constraint.py --epochs 10 --batch_size 16

# 4. 使用微调模型推理
python inference.py --model_path outputs/whisper_atc_constrained/final_model --use_processed

# 5. 评估改进效果
python evaluate_improvement.py --before outputs/final_model --after outputs/whisper_atc_constrained/final_model
```

## 💡 tips

1. **数据预处理**：确保已运行 `python train.py --preprocess_only`
2. **GPU显存**：微调需要最少8GB GPU显存（RTX3060或更高）
3. **训练时间**：完整数据集微调约需2-4小时
4. **验证**：每个epoch会自动在验证集上评估

## ❓ FAQ

**Q: 能否不重新训练，直接改进推理？**
A: 可以，但效果有限。词汇约束在训练时学到的约束知识会更强。

**Q: 需要多少训练数据？**
A: 至少1000条ATC语音样本。你现在有83个测试样本，建议用完整数据集训练。

**Q: 如何加入新的ATC词汇？**
A: 编辑 `atc_vocab/atc_vocab_config.yaml` 或直接修改 `build_atc_vocab.py` 中的关键词列表。

**Q: 约束是否会降低其他领域的泛化性？**
A: 不会。ATC约束是额外的损失项，不会删除原有的语言知识。

---

**准备好开始了吗？** 运行第一步：`python build_atc_vocab.py` ✨
