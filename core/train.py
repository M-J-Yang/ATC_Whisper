"""
Whisper 微调训练脚本（支持分布式 + fp16 + 缓存优化）
适配 RTX 3090 / HuggingFace Transformers >= 4.44
"""

import os
import json
import torch
import numpy as np
import argparse
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from evaluate import load as load_metric
from dataclasses import dataclass
from typing import Dict, Any, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 解析命令行参数
parser = argparse.ArgumentParser(description="Whisper 微调训练")
parser.add_argument("--unfreeze-encoder-layers", type=int, default=0,
                    help="解冻 encoder 的最后 N 层进行训练 (0=全部冻结)")
args = parser.parse_args()

# ======================================
# Data Collator for Speech Seq2Seq
# ======================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """数据整理器：动态 padding 音频特征和文本标签（特征已预提取）"""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 特征已经是 list 格式，需要转换为 tensor
        # input_features: (batch_size, 80, 3000) - mel spectrogram
        input_features = [torch.tensor(feature["input_features"]) for feature in features]
        label_features = [torch.tensor(feature["labels"]) for feature in features]

        # Pad 音频特征到相同长度
        batch = {}
        batch["input_features"] = torch.stack(input_features)

        # Pad 标签到相同长度
        max_label_length = max(len(l) for l in label_features)
        padded_labels = []
        for labels in label_features:
            padding_length = max_label_length - len(labels)
            if padding_length > 0:
                padded_labels.append(torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=labels.dtype)
                ]))
            else:
                padded_labels.append(labels)

        labels = torch.stack(padded_labels)

        # 如果所有序列都以 bos token 开头，移除它（Whisper 不需要）
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# ======================================
# ✅ 1. 加载配置
# ======================================
CONFIG_PATH = "./config.yaml"
import yaml
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用设备: {device}")

# ======================================
# ✅ 2. 加载模型与处理器
# ======================================
# 根据配置构建模型名称
model_type = config["model"]["type"]
if model_type == "whisper":
    whisper_size = config["model"]["whisper_size"]
    model_name = f"openai/whisper-{whisper_size}"
else:
    raise ValueError(f"不支持的模型类型: {model_type}")

logger.info(f"加载模型 {model_name}...")
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 冻结/解冻 encoder 层
if args.unfreeze_encoder_layers > 0:
    logger.info(f"冻结 encoder，仅解冻最后 {args.unfreeze_encoder_layers} 层")
    # 冻结所有 encoder 参数
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    # 解冻最后 N 层
    total_layers = len(model.model.encoder.layers)
    for i in range(total_layers - args.unfreeze_encoder_layers, total_layers):
        for param in model.model.encoder.layers[i].parameters():
            param.requires_grad = True
    logger.info(f"  - Encoder 总层数: {total_layers}")
    logger.info(f"  - 解冻层: {total_layers - args.unfreeze_encoder_layers} 到 {total_layers - 1}")
elif args.unfreeze_encoder_layers == 0:
    logger.info("冻结整个 encoder，仅训练 decoder")
    for param in model.model.encoder.parameters():
        param.requires_grad = False
else:
    logger.info("训练整个模型（encoder + decoder）")

model.to(device)

# ======================================
# ✅ 3. 加载数据集（从缓存，特征已预提取）
# ======================================
logger.info("从缓存加载数据集（特征已预提取）...")
# 使用 output_dir + processed_data 路径
dataset_path = os.path.join(config["output"]["output_dir"], "processed_data")
logger.info(f"数据集路径: {dataset_path}")
dataset = load_from_disk(dataset_path)

# 数据集已包含 input_features 和 labels，无需再处理
logger.info("✅ 数据集加载完成，特征已预提取")
logger.info(f"   - 训练集: {len(dataset['train'])} 条")
logger.info(f"   - 验证集: {len(dataset['val'])} 条")
logger.info(f"   - 测试集: {len(dataset['test'])} 条")

# ======================================
# ✅ 4. 定义评估指标
# ======================================
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    if isinstance(pred_ids, tuple):  # 🔧 修复 tuple 错误
        pred_ids = pred_ids[0]

    label_ids = pred.label_ids
    pred_ids = np.where(pred_ids == -100, processor.tokenizer.pad_token_id, pred_ids)
    label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ======================================
# ✅ 5. 训练参数
# ======================================
train_args = config["training"]
output_dir = config["output"]["output_dir"]

# 确保数值类型正确（从 YAML 读取可能是字符串）
learning_rate = float(train_args["learning_rate"])
weight_decay = float(train_args["weight_decay"])
batch_size = int(train_args["batch_size"])
gradient_accumulation_steps = int(train_args["gradient_accumulation_steps"])
warmup_steps = int(train_args["warmup_steps"])
num_train_epochs = int(train_args["epochs"])

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    warmup_steps=warmup_steps,
    num_train_epochs=num_train_epochs,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=100,
    save_total_limit=2,
    predict_with_generate=True,  # ✅ 必须为 True 才能在评估时生成文本
    fp16=True,
    gradient_checkpointing=False,
    dataloader_num_workers=int(config["system"]["num_workers"]),
    dataloader_pin_memory=bool(config["system"]["pin_memory"]),
    report_to="none",
    generation_max_length=225,
)

logger.info(f"训练配置:")
logger.info(f"  - Batch size: {batch_size}")
logger.info(f"  - Gradient accumulation: {gradient_accumulation_steps}")
logger.info(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
logger.info(f"  - Learning rate: {learning_rate}")
logger.info(f"  - Epochs: {num_train_epochs}")


# ======================================
# ✅ 6. 构建 Trainer
# ======================================
# 创建 data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    data_collator=data_collator,  # ✅ 使用正确的 data collator
    tokenizer=processor.tokenizer,  # ✅ 传入 tokenizer（用于保存）
    compute_metrics=compute_metrics,
)

# ======================================
# ✅ 7. 启动训练
# ======================================
logger.info("=" * 60)
logger.info("开始训练...")
logger.info("=" * 60)
trainer.train()

# ======================================
# ✅ 8. 保存模型
# ======================================
logger.info("训练完成，保存模型...")
final_model_path = os.path.join(config["output"]["output_dir"], "final_model")
trainer.save_model(final_model_path)
processor.save_pretrained(final_model_path)
logger.info(f"✅ 模型已保存到: {final_model_path}")
