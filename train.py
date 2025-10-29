"""
Whisper模型微调训练脚本 - 支持2张4090 GPU
使用HuggingFace Transformers和Accelerate进行分布式训练
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging
from dataclasses import dataclass
from functools import partial
import yaml
import librosa

import transformers
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup
)
from datasets import Dataset, DatasetDict, load_from_disk
from datasets import Audio
import evaluate
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2Seq:
    """用于Seq2Seq语音识别的数据整理器"""
    processor: WhisperProcessor

    def __call__(self, features):
        # features 已经包含预处理的 input_features 和 labels
        # 直接堆叠即可

        # 提取输入特征
        input_features = [{"input_features": item["input_features"]} for item in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # 处理标签
        label_features = [{"input_ids": item["labels"]} for item in features]

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # 用-100替换padding token ID，这样在损失计算中会被忽略
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        batch["labels"] = labels

        return batch


class WhisperTrainer:
    """Whisper模型训练器"""

    def __init__(self, config_path: str = "config.yaml"):
        """初始化训练器"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path(self.config["output"]["output_dir"])
        self.model_dir = Path(self.config["output"]["model_save_dir"])
        self.log_dir = Path(self.config["output"]["log_dir"])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"模型保存目录: {self.model_dir}")

    def load_dataset(self):
        """加载预处理的数据集"""
        processed_dir = self.output_dir / "processed_data"

        logger.info(f"从 {processed_dir} 加载数据集...")

        datasets_dict = {}
        for split in ["train", "val", "test"]:
            split_dir = processed_dir / split
            manifest_path = split_dir / f"{split}_manifest.json"

            if not manifest_path.exists():
                logger.error(f"找不到 {manifest_path}")
                raise FileNotFoundError(f"找不到 {manifest_path}")

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            # 转换为HuggingFace Dataset格式
            def gen():
                for item in manifest:
                    yield {
                        "audio": {
                            "path": item["audio_path"],
                            "array": None,
                            "sampling_rate": self.config["data"]["target_sr"]
                        },
                        "text": item["text"],
                        "speaker_id": item["speaker_id"],
                        "duration": item["duration"]
                    }

            dataset = Dataset.from_generator(gen)

            # 加载音频 - 使用librosa避免torchcodec依赖
            def load_audio(sample):
                try:
                    # 使用librosa加载，自动重采样到目标采样率
                    audio_array, sr = librosa.load(
                        sample["audio"]["path"],
                        sr=self.config["data"]["target_sr"],
                        mono=True
                    )
                except Exception as e:
                    logger.error(f"加载音频失败 {sample['audio']['path']}: {e}")
                    raise

                return {
                    "audio": {
                        "array": audio_array,
                        "sampling_rate": sr
                    },
                    "text": sample["text"],
                    "speaker_id": sample["speaker_id"],
                    "duration": sample["duration"]
                }

            dataset = dataset.map(
                load_audio,
                num_proc=self.config["system"]["num_workers"],
                desc=f"加载 {split} 音频"
            )

            datasets_dict[split] = dataset
            logger.info(f"{split}: {len(dataset)} 条样本")

        return DatasetDict(datasets_dict)

    def prepare_dataset(self, dataset: DatasetDict, processor: WhisperProcessor):
        """准备数据集特征"""
        logger.info("正在准备数据集特征...")

        def prepare_dataset_fn(batch):
            # 处理音频
            audio = batch["audio"]

            inputs = processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                language="en"
            )

            batch["input_features"] = inputs.input_features[0]
            batch["labels"] = processor.tokenizer(batch["text"]).input_ids

            return batch

        dataset = dataset.map(
            prepare_dataset_fn,
            remove_columns=["audio", "speaker_id"],
            num_proc=self.config["system"]["num_workers"],
            desc="准备特征"
        )

        return dataset

    def compute_metrics(self, pred, processor: WhisperProcessor):
        """计算WER (Word Error Rate)"""
        wer = evaluate.load("wer")
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # 解码预测和标签
        pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # 计算WER
        wer_score = wer.compute(predictions=pred_str, references=label_str)

        return {"wer": wer_score}

    def train(self, unfreeze_encoder_layers: int = 0, use_adapter: bool = False):
        """执行训练流程

        Args:
            unfreeze_encoder_layers: 解冻编码器的层数 (0=冻结, -1=全部)
            use_adapter: 是否使用Adapter层
        """
        logger.info("=" * 60)
        logger.info("开始Whisper模型微调")
        logger.info(f"参数: unfreeze_encoder_layers={unfreeze_encoder_layers}, use_adapter={use_adapter}")
        logger.info("=" * 60)

        # 1. 加载预训练模型和处理器
        model_name = f"openai/whisper-{self.config['model']['whisper_size']}"
        logger.info(f"加载模型: {model_name}")

        processor = WhisperProcessor.from_pretrained(model_name, language="English", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

        # 处理编码器冻结策略
        try:
            encoder = None
            if hasattr(model, 'encoder'):
                encoder = model.encoder
            elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
                encoder = model.model.encoder

            if encoder is None:
                logger.warning("无法找到编码器")
            else:
                if unfreeze_encoder_layers == -1:
                    # 解冻所有层
                    encoder.requires_grad_(True)
                    logger.info("✓ 解冻所有编码器层")
                elif unfreeze_encoder_layers > 0:
                    # 冻结所有层，然后解冻最后N层
                    encoder.requires_grad_(False)
                    encoder_layers = list(encoder.children())
                    num_layers = len(encoder_layers)
                    unfreeze_from = max(0, num_layers - unfreeze_encoder_layers)

                    for i, layer in enumerate(encoder_layers):
                        if i >= unfreeze_from:
                            layer.requires_grad_(True)
                    logger.info(f"✓ 解冻最后 {unfreeze_encoder_layers} 层编码器 (共 {num_layers} 层)")
                else:
                    # 冻结所有编码器层
                    encoder.requires_grad_(False)
                    logger.info("✓ 冻结所有编码器层")
        except Exception as e:
            logger.warning(f"处理编码器失败: {e}，将对整个模型进行微调")

        # 2. 加载数据集
        datasets = self.load_dataset()

        # 3. 准备数据集
        prepared_datasets = self.prepare_dataset(datasets, processor)

        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA可用: {cuda_available}")

        if cuda_available:
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
            logger.info(f"当前GPU: {torch.cuda.get_device_name(0)}")

        # 4. 定义训练参数
        # 注意：FP16在Windows或某些GPU上可能有兼容性问题，禁用FP16以确保稳定性
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.model_dir),
            per_device_train_batch_size=self.config["training"]["batch_size"],
            per_device_eval_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=float(self.config["training"]["learning_rate"]),  # 确保是浮点数
            num_train_epochs=self.config["training"]["epochs"],
            warmup_steps=self.config["training"]["warmup_steps"],
            weight_decay=self.config["training"]["weight_decay"],

            # 多GPU配置
            ddp_find_unused_parameters=False,
            ddp_backend="nccl" if (self.config["training"]["distributed"] and cuda_available) else None,

            # 评估和保存策略
            evaluation_strategy="steps",
            eval_steps=self.config["training"]["eval_steps"],
            save_strategy="steps",
            save_steps=self.config["training"]["save_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            save_total_limit=3,

            # 优化器
            optim="adamw_torch",
            max_grad_norm=self.config["system"]["max_grad_norm"],

            # 日志
            logging_steps=100,
            logging_dir=str(self.log_dir),
            report_to="tensorboard",

            # 混合精度 - 在Windows上禁用FP16以避免兼容性问题
            # 使用标准精度(FP32)训练，但显存占用更多
            fp16=False,
            bf16=False,

            # 推送到Hub
            push_to_hub=False,
            seed=self.config["system"]["seed"],
        )

        logger.info("训练参数: FP16=False, BF16=False (标准FP32精度)")

        # 5. 初始化数据整理器
        data_collator = DataCollatorSpeechSeq2Seq(processor=processor)

        # 6. 创建训练器
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=prepared_datasets["train"],
            eval_dataset=prepared_datasets["val"],
            data_collator=data_collator,
            compute_metrics=partial(self.compute_metrics, processor=processor),
            tokenizer=processor.tokenizer,
        )

        # 7. 开始训练
        logger.info("开始训练...")
        trainer.train()

        # 8. 保存最终模型
        logger.info(f"保存最终模型到 {self.model_dir}")
        trainer.save_model(str(self.model_dir / "final_model"))
        processor.save_pretrained(str(self.model_dir / "final_model"))

        logger.info("=" * 60)
        logger.info("训练完成！")
        logger.info(f"最佳模型: {self.model_dir}")
        logger.info("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Whisper模型微调训练")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--unfreeze-encoder-layers", type=int, default=0,
                        help="解冻最后N层编码器 (0=仅微调解码器, -1=所有层)")
    parser.add_argument("--use-adapter", action="store_true", help="使用Adapter层替代全量微调")

    args = parser.parse_args()

    trainer = WhisperTrainer(config_path=args.config)

    # 如果指定了unfreeze层数，修改模型冻结策略
    if args.unfreeze_encoder_layers != 0:
        logger.info(f"将解冻编码器的最后 {args.unfreeze_encoder_layers} 层")
        # 这个在train()方法中实现

    trainer.train(unfreeze_encoder_layers=args.unfreeze_encoder_layers, use_adapter=args.use_adapter)


if __name__ == "__main__":
    main()
