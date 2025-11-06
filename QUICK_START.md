# 快速开始指南

## 🎯 选择使用方式

### 💻 Web 应用（推荐 - 最简单）

**适合**：想要可视化界面、实时识别、查看统计数据

**步骤**：
1. 安装依赖：`.\install_dependencies.ps1`
2. 一键启动：`.\start_all.ps1`
3. 浏览器访问：http://localhost:3000
4. 点击"加载模型"
5. 开始推理！

---

### ⌨️ 命令行（推荐 - 最快速）

**适合**：批量处理、自动化、服务器环境

**单次推理**：
```bash
# 1. 配置音频路径（编辑 config.yaml）
single_wav:
  path: "your_audio.wav"

# 2. 运行
python scripts/inference_single.py
```

**交互式推理**：
```bash
python scripts/inference_interactive.py
# 然后输入音频路径进行连续推理
```

---

### 🔬 训练模型（高级）

**适合**：需要自定义训练、提升精度

**步骤**：
```bash
# 1. 数据预处理
python core/preprocess.py

# 2. 训练
python core/train.py

# 3. 评估
python core/inference.py \
  --model_path models/final_model \
  --dataset_dir processed_data \
  --split test
```

---

## 📁 重要文件

| 文件 | 说明 |
|------|------|
| `config.yaml` | 全局配置 |
| `PROJECT_STRUCTURE.md` | 项目结构详解 |
| `README.md` | 完整文档 |
| `README_WEB.md` | Web 应用说明 |

---

## 🐛 常见问题

### Q: 推理慢？
A: 首次推理需要 GPU 预热（~1.6秒），之后会很快（~0.36秒）

### Q: 模型在哪？
A: 训练后在 `models/final_model/`，或使用预训练模型 `openai/whisper-base`

### Q: 如何修改配置？
A: 编辑 `config.yaml`

### Q: Web 界面打不开？
A: 检查后端是否启动（http://localhost:8000），前端是否启动（http://localhost:3000）

---

## 🚀 下一步

- [查看项目结构](PROJECT_STRUCTURE.md)
- [阅读完整文档](README.md)
- [Web 应用文档](README_WEB.md)
