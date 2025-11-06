# 项目文件结构（清理后）

## 📂 根目录文件

```
demo/
├── config.yaml                    # 全局配置文件
├── requirements.txt               # Python 依赖
├── .gitignore                     # Git 忽略配置
│
├── README.md                      # 📖 主文档
├── README_WEB.md                  # 📖 Web 应用文档
├── PROJECT_STRUCTURE.md           # 📖 项目结构说明
├── QUICK_START.md                 # 📖 快速开始指南
│
├── start_all.ps1                  # 🚀 一键启动（Web）
├── install_dependencies.ps1       # 📦 安装所有依赖
├── start_backend.bat              # 🔧 启动后端
├── start_frontend.bat             # 🎨 启动前端
└── gitpush.ps1                    # Git 提交脚本
```

## 📁 核心模块 (core/)

```
core/
├── inference.py                   # Whisper 推理引擎
├── preprocess.py                  # 数据预处理
├── train.py                       # 模型训练
└── atc_decoder.py                 # ATC 词汇约束解码器
```

**用途**：核心算法实现，独立于 Web 应用

## 📁 后端服务 (backend/)

```
backend/
├── app.py                         # FastAPI 主应用
├── inference_service.py           # 推理服务（单例模式）
├── requirements.txt               # 后端专用依赖
└── uploads/                       # 上传文件临时目录
```

**用途**：提供 REST API 和 WebSocket 服务

## 📁 前端应用 (frontend/)

```
frontend/
├── public/
│   └── index.html                 # HTML 模板
├── src/
│   ├── components/                # React 组件
│   │   ├── ModelConfig.js         # 模型配置
│   │   ├── SingleInference.js     # 单条推理
│   │   ├── RealtimeRecognition.js # 实时识别
│   │   └── ResultsLog.js          # 结果记录
│   ├── services/
│   │   └── api.js                 # API 服务封装
│   ├── App.js                     # 主应用
│   ├── App.css
│   ├── index.js
│   └── index.css
└── package.json                   # Node.js 依赖
```

**用途**：Web 用户界面

## 📁 命令行工具 (scripts/)

```
scripts/
├── inference_single.py            # 单条推理（读取 config.yaml）
└── inference_interactive.py       # 交互式推理（连续推理）
```

**用途**：命令行批量处理工具

## 📁 数据目录

```
models/                            # 训练好的模型
├── final_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...

outputs/                           # 输出结果
├── inference_results/
└── training_logs/

processed_data/                    # 预处理数据
├── train/
├── val/
└── test/

ATCOSIM/                          # 原始数据集
├── WAVdata/
└── TXTdata/
    ├── fulldata.csv
    └── wordlist.txt

logs/                             # 日志文件
```

## 🎯 使用场景

### 1. Web 应用
```
启动： start_all.ps1
访问： http://localhost:3000
```

### 2. 命令行推理
```bash
python scripts/inference_single.py      # 单次
python scripts/inference_interactive.py  # 交互式
```

### 3. 训练模型
```bash
python core/preprocess.py  # 预处理
python core/train.py       # 训练
python core/inference.py   # 评估
```

## 📊 文件统计

| 类型 | 数量 |
|------|------|
| 核心模块 | 4 个 |
| 后端文件 | 2 个 |
| 前端组件 | 4 个 |
| 命令行工具 | 2 个 |
| 文档 | 4 个 |
| 启动脚本 | 4 个 |

## ✅ 清理说明

已删除以下重复文件：
- ❌ 主目录下的 `inference.py`（已移到 core/）
- ❌ 主目录下的 `preprocess.py`（已移到 core/）
- ❌ 主目录下的 `train.py`（已移到 core/）
- ❌ 主目录下的 `atc_decoder.py`（已移到 core/）
- ❌ 主目录下的 `inference_single.py`（已移到 scripts/）
- ❌ 主目录下的 `inference_interactive.py`（已移到 scripts/）
- ❌ 主目录下的 `inference_service.py`（已移到 backend/）
- ❌ `test_inference_speed.py`（测试文件）
- ❌ `single_inference_result.txt`（临时文件）
- ❌ `calude_database.md`（无关文件）
- ❌ `QUICKSTART.md`（旧版文档，已有 QUICK_START.md）

## 🔄 导入路径

所有模块使用统一导入方式：

```python
# 在 backend/ 或 scripts/ 中导入 core 模块
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.inference import WhisperInference
from core.atc_decoder import ATCVocabularyConstraint
```

## 📝 注意事项

1. **不要在主目录创建新的 .py 文件**
2. **核心功能放 core/**
3. **Web API 放 backend/**
4. **命令行工具放 scripts/**
5. **文档统一放根目录**
