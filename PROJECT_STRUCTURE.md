# 项目结构说明

```
demo/
│
├── 📁 core/                          # 核心模块
│   ├── inference.py                  # Whisper 推理引擎
│   ├── preprocess.py                 # 数据预处理
│   ├── train.py                      # 模型训练
│   └── atc_decoder.py                # ATC 词汇约束解码器
│
├── 📁 backend/                       # Web 后端服务
│   ├── app.py                        # FastAPI 主应用
│   ├── inference_service.py          # 推理服务（单例）
│   ├── requirements.txt              # Python 依赖
│   └── uploads/                      # 上传文件临时目录
│
├── 📁 frontend/                      # Web 前端应用
│   ├── public/
│   │   └── index.html                # HTML 模板
│   ├── src/
│   │   ├── components/               # React 组件
│   │   │   ├── ModelConfig.js        # 模型配置组件
│   │   │   ├── SingleInference.js    # 单条推理组件
│   │   │   ├── RealtimeRecognition.js # 实时识别组件
│   │   │   └── ResultsLog.js         # 结果记录组件
│   │   ├── services/
│   │   │   └── api.js                # API 服务
│   │   ├── App.js                    # 主应用
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   └── package.json                  # Node.js 依赖
│
├── 📁 scripts/                       # 命令行脚本
│   ├── inference_single.py           # 单条推理脚本
│   ├── inference_interactive.py      # 交互式推理脚本
│   └── utils.py                      # 工具函数
│
├── 📁 models/                        # 模型文件
│   └── final_model/                  # 训练好的模型
│
├── 📁 outputs/                       # 输出目录
│   ├── inference_results/            # 推理结果
│   └── training_logs/                # 训练日志
│
├── 📁 processed_data/                # 预处理数据
│   ├── train/
│   ├── val/
│   └── test/
│
├── 📁 ATCOSIM/                       # 原始数据集
│   ├── WAVdata/                      # 音频文件
│   └── TXTdata/                      # 文本数据
│       ├── fulldata.csv
│       └── wordlist.txt
│
├── 📁 logs/                          # 日志文件
│
├── 📄 config.yaml                    # 全局配置文件
│
├── 📄 README.md                      # 项目说明（主）
├── 📄 README_WEB.md                  # Web 应用说明
│
├── 🚀 start_all.ps1                  # 一键启动（Web）
├── 🔧 install_dependencies.ps1       # 安装依赖
├── 📦 requirements.txt               # 项目依赖（总）
│
└── 📄 .gitignore                     # Git 忽略配置
```

## 📚 模块说明

### Core 模块（核心）
- **inference.py**：Whisper 推理引擎，提供模型推理、评估功能
- **preprocess.py**：数据预处理，音频特征提取
- **train.py**：模型训练脚本

### Backend 模块（Web 后端）
- **app.py**：FastAPI 服务，提供 REST API 和 WebSocket
- **inference_service.py**：推理服务管理，单例模式，处理模型加载和预热

### Frontend 模块（Web 前端）
- **components/**：React 组件
  - ModelConfig：模型配置界面
  - SingleInference：单条推理界面
  - RealtimeRecognition：实时识别界面
  - ResultsLog：结果记录表格
- **services/api.js**：API 调用封装

### Scripts 模块（命令行工具）
- **inference_single.py**：单次推理脚本（读取 config.yaml）
- **inference_interactive.py**：交互式推理脚本（连续推理）

## 🔄 使用场景

### 1. 命令行推理
```bash
# 单次推理
python scripts/inference_single.py

# 交互式推理
python scripts/inference_interactive.py
```

### 2. Web 应用
```bash
# 一键启动
.\start_all.ps1

# 或分别启动
python backend/app.py
cd frontend && npm start
```

### 3. 训练模型
```bash
python core/train.py
```

### 4. 数据预处理
```bash
python core/preprocess.py
```

## 📦 依赖安装

### 后端依赖
```bash
pip install -r backend/requirements.txt
```

### 前端依赖
```bash
cd frontend
npm install
```

## 🔧 配置文件

所有配置统一在 `config.yaml` 中管理：
- 数据路径配置
- 模型配置
- 训练参数
- 推理参数
- 单条推理配置
