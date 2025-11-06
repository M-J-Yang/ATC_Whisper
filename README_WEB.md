# ATC 语音识别 Web 应用

基于 React + FastAPI 的语音识别 Web 应用系统。

## 📁 项目结构

```
demo/
├── backend/                 # 后端服务
│   ├── app.py              # FastAPI 主应用
│   ├── api/                # API 路由
│   └── requirements.txt    # Python 依赖
├── frontend/               # 前端应用
│   ├── public/             # 静态文件
│   ├── src/
│   │   ├── components/     # React 组件
│   │   ├── services/       # API 服务
│   │   ├── App.js          # 主应用
│   │   └── index.js        # 入口文件
│   └── package.json        # Node.js 依赖
├── inference_service.py    # 推理服务
├── config.yaml             # 配置文件
└── README_WEB.md           # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

#### 后端依赖
```bash
cd backend
pip install -r requirements.txt
```

#### 前端依赖
```bash
cd frontend
npm install
```

### 2. 启动服务

#### 启动后端服务（终端1）
```bash
cd backend
python app.py
```
后端服务将在 http://localhost:8000 启动

#### 启动前端服务（终端2）
```bash
cd frontend
npm start
```
前端应用将在 http://localhost:3000 自动打开

### 3. 使用应用

1. **加载模型**：点击"加载模型"按钮，等待模型加载完成（首次需要预热 GPU）
2. **单条推理**：上传音频文件，点击"开始推理"
3. **实时识别**：点击"开始实时识别"，允许麦克风权限，开始录音识别
4. **查看结果**：所有推理结果会实时显示在下方表格中
5. **导出结果**：点击"导出"按钮，将结果保存为 CSV 文件

## ✨ 功能特性

### 1. 模型配置
- ✅ 一键加载模型
- ✅ 显示模型配置信息
- ✅ GPU 预热优化
- ✅ 自动检测模型状态

### 2. 单条语音推理
- ✅ 支持多种音频格式（wav, mp3, flac, m4a）
- ✅ 拖拽上传
- ✅ 实时显示推理进度
- ✅ 词汇约束支持

### 3. 实时语音识别
- ✅ 浏览器麦克风录音
- ✅ 每 2 秒自动识别
- ✅ WebSocket 实时通信
- ✅ 实时结果显示

### 4. 结果记录
- ✅ 记录所有推理结果
- ✅ 显示推理时间、RTF 等指标
- ✅ 系统时间戳记录
- ✅ 导出为 CSV 格式
- ✅ 分页显示

## 🔧 API 接口

### 配置相关
- `POST /api/config/load` - 加载模型配置
- `GET /api/config/status` - 获取配置状态

### 推理相关
- `POST /api/inference/single` - 单条语音推理
- `WS /ws/realtime` - 实时语音识别（WebSocket）

### 工具相关
- `DELETE /api/uploads/clear` - 清理上传文件

## 📊 推理结果字段

| 字段 | 说明 |
|------|------|
| type | 推理类型（single/realtime） |
| filename | 文件名 |
| text | 识别文本 |
| duration | 音频时长（秒） |
| inference_time | 推理耗时（秒） |
| total_time | 总耗时（秒） |
| rtf | 实时率（Real-Time Factor） |
| timestamp | 显示时间 |
| system_time | 系统时间（ISO格式） |

## ⚙️ 配置说明

在 `config.yaml` 中配置：

```yaml
single_wav:
  model_path: "models/final_model"  # 模型路径
  use_vocab_constraint: true         # 启用词汇约束

data:
  wordlist_path: "ATCOSIM/TXTdata/wordlist.txt"  # 词表路径

inference:
  language: "en"                     # 语言
  max_length: 225                    # 最大生成长度
```

## 🐛 故障排除

### 后端无法启动
- 检查 Python 环境和依赖安装
- 确认 8000 端口未被占用
- 查看 `inference_service.py` 是否在正确位置

### 前端无法连接后端
- 确认后端服务已启动
- 检查 CORS 配置
- 查看浏览器控制台错误信息

### 实时识别无声音
- 检查麦克风权限
- 确认浏览器支持 MediaRecorder API
- 使用 Chrome/Edge 浏览器

### 推理速度慢
- 首次推理需要预热 GPU（约 1.6 秒）
- 后续推理应该在 0.3-0.4 秒左右
- 检查 GPU 是否正常工作

## 📝 开发说明

### 后端开发
```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 前端开发
```bash
cd frontend
npm start
```

### 构建生产版本
```bash
cd frontend
npm run build
```

## 🎯 技术栈

### 后端
- FastAPI - Web 框架
- Uvicorn - ASGI 服务器
- WebSocket - 实时通信
- PyTorch - 深度学习
- Transformers - 预训练模型

### 前端
- React 18 - UI 框架
- Ant Design - UI 组件库
- Axios - HTTP 客户端
- WebSocket - 实时通信
- Day.js - 时间处理

## 📄 许可证

MIT License

## 👥 贡献者

- Your Name
