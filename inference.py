"""
语音识别推理脚本 - 支持单个音频和批量推理
包含WER评估、ATC词汇约束等功能
"""

import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import yaml
import csv

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperInference:
    """Whisper推理引擎"""

    def __init__(self, model_path: str, config_path: str = "config.yaml", device: str = "cuda"):
        """初始化推理引擎

        Args:
            model_path: 模型目录路径
            config_path: 配置文件路径
            device: 推理设备 (cuda, cpu)
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)

        logger.info(f"从 {model_path} 加载模型...")
        logger.info(f"使用设备: {self.device}")

        # 加载处理器和模型
        self.processor = WhisperProcessor.from_pretrained(str(self.model_path))
        self.model = WhisperForConditionalGeneration.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

        logger.info("模型加载完成")

    def transcribe_file(self, audio_path: str, language: str = "en", vocab_constraint: Optional[set] = None) -> Dict[str, any]:
        """转录单个音频文件

        Args:
            audio_path: 音频文件路径
            language: 语言代码 (en, zh, etc.)
            vocab_constraint: 词表约束集合（仅允许这些词出现）

        Returns:
            {
                "text": "转录文本",
                "confidence": 0.95,
                "duration": 3.5
            }
        """
        try:
            # 加载音频
            audio_array, sr = torchaudio.load(audio_path)
            audio_array = audio_array.squeeze().numpy()

            # 创建输入特征
            inputs = self.processor(
                audio_array,
                sampling_rate=sr,
                language=language,
                return_tensors="pt"
            )

            input_features = inputs.input_features.to(self.device)

            # 推理
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    language=language,
                    task="transcribe",
                    max_new_tokens=self.config["inference"]["max_length"],
                    temperature=self.config["inference"]["temperature"],
                    num_beams=self.config["inference"]["beam_size"]
                )

            # 解码
            text = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            # 应用词表约束（后处理）
            if vocab_constraint:
                text = self._apply_vocab_constraint(text, vocab_constraint)

            # 获取音频时长
            duration = len(audio_array) / sr

            return {
                "text": text.strip(),
                "duration": duration,
                "audio_path": audio_path
            }

        except Exception as e:
            logger.error(f"转录失败 {audio_path}: {e}")
            return {
                "text": "",
                "duration": 0,
                "audio_path": audio_path,
                "error": str(e)
            }

    def _apply_vocab_constraint(self, text: str, vocab_constraint: set) -> str:
        """应用词表约束 - 将不在约束词表中的词替换为<unk>

        Args:
            text: 转录文本
            vocab_constraint: 允许的词汇集合

        Returns:
            约束后的文本
        """
        words = text.lower().split()
        constrained_words = []

        for word in words:
            # 检查词是否在约束词表中
            if word in vocab_constraint:
                constrained_words.append(word)
            else:
                # 尝试部分匹配（处理标点符号）
                clean_word = "".join(c for c in word if c.isalnum())
                if clean_word in vocab_constraint:
                    constrained_words.append(clean_word)
                else:
                    constrained_words.append("<unk>")

        return " ".join(constrained_words)

    def transcribe_batch(self, audio_paths: List[str], language: str = "en") -> List[Dict]:
        """批量转录音频文件

        Args:
            audio_paths: 音频文件路径列表
            language: 语言代码

        Returns:
            转录结果列表
        """
        results = []

        logger.info(f"开始批量转录 {len(audio_paths)} 个文件...")

        for audio_path in tqdm(audio_paths):
            result = self.transcribe_file(audio_path, language)
            results.append(result)

        return results

    def transcribe_dataset(self, dataset_dir: str, split: str = "test") -> List[Dict]:
        """转录整个数据集

        Args:
            dataset_dir: 数据集目录
            split: 数据集划分 (train, val, test)

        Returns:
            转录结果列表
        """
        split_dir = Path(dataset_dir) / split
        manifest_path = split_dir / f"{split}_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"找不到 {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        audio_paths = [item["audio_path"] for item in manifest]

        logger.info(f"转录 {split} 数据集 ({len(audio_paths)} 条样本)...")

        results = self.transcribe_batch(audio_paths)

        # 添加参考文本和speaker_id
        for i, result in enumerate(results):
            result["reference_text"] = manifest[i]["text"]
            result["speaker_id"] = manifest[i]["speaker_id"]

        return results


class WhisperEvaluator:
    """Whisper模型评估器"""

    def __init__(self, config_path: str = "config.yaml"):
        """初始化评估器"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

        logger.info("评估器初始化完成")

    def compute_wer(self, predictions: List[str], references: List[str]) -> float:
        """计算词错率 (Word Error Rate)"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考文本数量不匹配")

        wer_score = self.wer_metric.compute(
            predictions=predictions,
            references=references
        )

        return wer_score

    def compute_cer(self, predictions: List[str], references: List[str]) -> float:
        """计算字错率 (Character Error Rate)"""
        if len(predictions) != len(references):
            raise ValueError("预测和参考文本数量不匹配")

        cer_score = self.cer_metric.compute(
            predictions=predictions,
            references=references
        )

        return cer_score

    def evaluate_results(self, results: List[Dict]) -> Dict[str, float]:
        """评估转录结果

        Args:
            results: 转录结果列表，包含 "text" 和 "reference_text" 字段

        Returns:
            评估指标字典
        """
        predictions = [r["text"] for r in results if "text" in r]
        references = [r["reference_text"] for r in results if "reference_text" in r]

        logger.info(f"评估 {len(predictions)} 条样本...")

        # 计算WER
        wer_score = self.compute_wer(predictions, references)

        # 计算CER
        cer_score = self.compute_cer(predictions, references)

        # 按说话人计算WER
        speaker_wers = {}
        if "speaker_id" in results[0]:
            speakers = {}
            for result in results:
                speaker_id = result.get("speaker_id", "unknown")
                if speaker_id not in speakers:
                    speakers[speaker_id] = {"pred": [], "ref": []}

                speakers[speaker_id]["pred"].append(result.get("text", ""))
                speakers[speaker_id]["ref"].append(result.get("reference_text", ""))

            for speaker_id, texts in speakers.items():
                wer = self.compute_wer(texts["pred"], texts["ref"])
                speaker_wers[speaker_id] = wer

        metrics = {
            "wer": wer_score,
            "cer": cer_score,
            "num_samples": len(predictions),
            "speaker_wers": speaker_wers
        }

        return metrics

    def print_results(self, results: List[Dict], max_samples: int = 10):
        """打印转录样本"""
        logger.info(f"显示前 {min(max_samples, len(results))} 条样本:")
        logger.info("=" * 100)

        for i, result in enumerate(results[:max_samples]):
            logger.info(f"\n样本 {i+1}:")
            logger.info(f"参考: {result.get('reference_text', 'N/A')}")
            logger.info(f"预测: {result.get('text', 'N/A')}")
            if "speaker_id" in result:
                logger.info(f"说话人: {result['speaker_id']}")

        logger.info("\n" + "=" * 100)

    def save_results(self, results: List[Dict], output_path: str):
        """保存转录结果到CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if not results:
                return

            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

        logger.info(f"结果已保存到 {output_path}")

    def generate_report(self, results: List[Dict], metrics: Dict, output_dir: str):
        """生成评估报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON报告
        report = {
            "metrics": metrics,
            "results": results[:100]  # 只保存前100条用于查看
        }

        report_path = output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 保存详细结果CSV
        csv_path = output_dir / "transcription_results.csv"
        self.save_results(results, str(csv_path))

        # 打印摘要
        logger.info("\n" + "=" * 60)
        logger.info("评估摘要")
        logger.info("=" * 60)
        logger.info(f"总样本数: {metrics['num_samples']}")
        logger.info(f"词错率 (WER): {metrics['wer']:.2%}")
        logger.info(f"字错率 (CER): {metrics['cer']:.2%}")

        if metrics.get("speaker_wers"):
            logger.info("\n按说话人的WER:")
            for speaker_id, wer in sorted(metrics["speaker_wers"].items()):
                logger.info(f"  {speaker_id}: {wer:.2%}")

        logger.info(f"\n报告已保存到 {output_dir}")


def main():
    """主函数 - 演示推理和评估"""
    import argparse

    parser = argparse.ArgumentParser(description="Whisper推理和评估")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--audio_path", type=str, help="单个音频文件路径")
    parser.add_argument("--dataset_dir", type=str, help="数据集目录")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="d:\\NPU_works\\语音\\demo\\outputs\\results")
    parser.add_argument("--vocab_constraint", type=str, help="ATC词表文件路径（用于词表约束）")

    args = parser.parse_args()

    # 加载词表约束（可选）
    vocab_constraint = None
    if args.vocab_constraint and Path(args.vocab_constraint).exists():
        with open(args.vocab_constraint, "r", encoding="utf-8") as f:
            vocab_constraint = {line.strip().lower() for line in f if line.strip()}
        logger.info(f"加载词表约束: {len(vocab_constraint)} 个词")

    # 初始化推理引擎
    inference_engine = WhisperInference(args.model_path)

    # 初始化评估器
    evaluator = WhisperEvaluator()

    # 推理
    if args.audio_path:
        # 转录单个文件
        result = inference_engine.transcribe_file(args.audio_path, vocab_constraint=vocab_constraint)
        logger.info(f"转录结果: {result['text']}")

    elif args.dataset_dir:
        # 转录整个数据集
        results = inference_engine.transcribe_dataset(args.dataset_dir, split=args.split)

        # 评估
        metrics = evaluator.evaluate_results(results)

        # 显示和保存结果
        evaluator.print_results(results)
        evaluator.generate_report(results, metrics, args.output_dir)


if __name__ == "__main__":
    main()
