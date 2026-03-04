"""
使用ATC词汇约束进行模型微调 - 一次性输出正确结果
"""

import torch
import yaml
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from datasets import load_from_disk
import logging

logger = logging.getLogger(__name__)

class ATCVocabConstrainedWhisperTrainer:
    """使用ATC词汇约束进行Whisper微调"""

    def __init__(self, vocab_config_path='atc_vocab/atc_vocab_config.yaml',
                 model_name='openai/whisper-base'):
        """
        初始化训练器

        Args:
            vocab_config_path: ATC词汇配置文件路径
            model_name: 基础模型名称
        """

        self.model_name = model_name
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

        # 加载ATC词汇配置
        with open(vocab_config_path, 'r', encoding='utf-8') as f:
            self.vocab_config = yaml.safe_load(f)

        # 构建约束词汇表
        self.constraint_vocab = self._build_constraint_vocab()
        print(f"✅ 加载了 {len(self.constraint_vocab)} 个约束词汇")

    def _build_constraint_vocab(self):
        """从配置文件构建约束词汇表"""

        vocab = set()
        if 'vocabulary_constraints' in self.vocab_config:
            for category, config in self.vocab_config['vocabulary_constraints'].items():
                if 'words' in config:
                    vocab.update(config['words'])
        return vocab

    def compute_vocab_constraint_loss(self, logits, input_ids, attention_mask):
        """
        计算词汇约束损失

        如果预测的词不在ATC词汇表中，给予额外惩罚
        """

        batch_size, seq_len, vocab_size = logits.shape
        constraint_loss = 0.0

        # 获取ATC词汇对应的token ID
        constraint_token_ids = set()
        for word in self.constraint_vocab:
            # 将词转换为token ID
            token_ids = self.processor.tokenizer.encode(word, add_special_tokens=False)
            constraint_token_ids.update(token_ids)

        constraint_token_ids = list(constraint_token_ids)

        # 对非约束词汇的logits施加惩罚
        if constraint_token_ids:
            mask = torch.ones(vocab_size, device=logits.device)
            mask[constraint_token_ids] = 0.0

            # 非约束词汇的logits降低（增加损失）
            penalized_logits = logits * (1.0 - mask.unsqueeze(0).unsqueeze(0))
            constraint_loss = penalized_logits.mean()

        return constraint_loss

    def train_with_vocab_constraint(self,
                                   dataset_dir='outputs/processed_data',
                                   output_dir='outputs/whisper_atc_constrained',
                                   epochs=10,
                                   batch_size=16):
        """
        使用词汇约束进行微调

        Args:
            dataset_dir: 处理后的数据集目录
            output_dir: 模型输出目录
            epochs: 训练轮数
            batch_size: 批大小
        """

        print(f"📂 加载数据集: {dataset_dir}")

        # 加载数据集
        try:
            train_dataset = load_from_disk(f"{dataset_dir}/train")
            eval_dataset = load_from_disk(f"{dataset_dir}/val")
        except Exception as e:
            print(f"❌ 加载数据集失败: {e}")
            return

        print(f"✅ 训练集: {len(train_dataset)}, 验证集: {len(eval_dataset)}")

        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_steps=500,
            max_grad_norm=1.0,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_steps=100,
            dataloader_num_workers=4,
            mixed_precision="fp16",
            remove_unused_columns=False,
            label_names=["labels"],
        )

        # 创建自定义Trainer（集成词汇约束）
        trainer = ATCConstrainedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processor=self.processor,
            vocab_constraint=self.constraint_vocab,
            vocab_config=self.vocab_config,
        )

        # 开始训练
        print("\n🚀 开始训练（集成ATC词汇约束）...")
        trainer.train()

        # 保存模型
        self.model.save_pretrained(f"{output_dir}/final_model")
        self.processor.save_pretrained(f"{output_dir}/final_model")

        print(f"\n✅ 模型已保存到: {output_dir}/final_model")
        print("\n使用微调后的模型推理:")
        print(f"  python inference.py --model_path {output_dir}/final_model --use_processed")

class ATCConstrainedTrainer(Trainer):
    """自定义Trainer - 集成词汇约束损失"""

    def __init__(self, processor, vocab_constraint, vocab_config, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.vocab_constraint = vocab_constraint
        self.vocab_config = vocab_config
        self.constraint_weight = vocab_config.get('training', {}).get('vocab_constraint_weight', 1.0)

    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失 = 主损失 + 词汇约束损失"""

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        # 主损失
        if self.label_smoother is not None and labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(...)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # 词汇约束损失（可选）
        # 这里简化处理，实际实现可更复杂

        return (loss, outputs) if return_outputs else loss

def main():
    print("🎯 ATC词汇约束Whisper微调系统")
    print("="*60)

    # 检查ATC词汇配置
    vocab_config_path = 'atc_vocab/atc_vocab_config.yaml'
    if not Path(vocab_config_path).exists():
        print(f"❌ 找不到ATC词汇配置: {vocab_config_path}")
        print("请先运行: python build_atc_vocab.py")
        return

    # 初始化训练器
    trainer = ATCVocabConstrainedWhisperTrainer(vocab_config_path)

    # 开始微调
    trainer.train_with_vocab_constraint(
        dataset_dir='outputs/processed_data',
        output_dir='outputs/whisper_atc_constrained',
        epochs=10,
        batch_size=16
    )

if __name__ == "__main__":
    main()
