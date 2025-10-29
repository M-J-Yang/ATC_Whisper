#!/bin/bash
# ATCOSIM 语音识别完整训练流程脚本

set -e

echo "=================================="
echo "ATCOSIM 语音识别系统"
echo "双GPU (4090 x2) 训练管道"
echo "=================================="

# 配置
CONFIG_PATH="config.yaml"
DATASET_DIR="d:\NPU_works\语音\demo\ATCOSIM"
OUTPUT_DIR="d:\NPU_works\语音\demo\outputs"

echo ""
echo "【步骤 1】检查环境和安装依赖"
echo "=================================="

if ! command -v python &> /dev/null; then
    echo "❌ Python 未安装!"
    exit 1
fi

python --version

echo "安装依赖库..."
pip install -r requirements.txt -q

echo ""
echo "【步骤 2】数据预处理"
echo "=================================="

if [ -d "$OUTPUT_DIR/processed_data" ]; then
    echo "✓ 已有处理过的数据，跳过预处理"
else
    echo "正在处理 ATCOSIM 数据集..."
    python preprocess.py
fi

echo ""
echo "【步骤 3】模型训练"
echo "=================================="

echo "使用以下配置开始训练:"
echo "- 模型: Whisper-base"
echo "- GPU: 2张 NVIDIA 4090"
echo "- Batch Size: 8 (每张GPU 4)"
echo "- 学习率: 1e-5"
echo "- Epochs: 10"
echo ""

read -p "确认开始训练? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train.py
else
    echo "训练已取消"
fi

echo ""
echo "【步骤 4】推理和评估"
echo "=================================="

MODEL_PATH="$OUTPUT_DIR/models/final_model"

if [ -d "$MODEL_PATH" ]; then
    echo "使用模型进行推理..."
    python inference.py \
        --model_path "$MODEL_PATH" \
        --dataset_dir "$OUTPUT_DIR/processed_data" \
        --split test \
        --output_dir "$OUTPUT_DIR/results"
else
    echo "⚠ 未找到训练好的模型"
fi

echo ""
echo "=================================="
echo "✓ 流程完成!"
echo "=================================="
echo "结果位置: $OUTPUT_DIR"
echo "模型位置: $MODEL_PATH"
