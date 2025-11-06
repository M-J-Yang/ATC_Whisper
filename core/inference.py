"""
Whisper 高效推理与评估脚本（单条推理优化版）
支持单音频 / 批量数据集 / WER评估 / ATC词汇约束
"""

import os
import json
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import yaml
import csv
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from datasets import load_from_disk 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperInference:
    """Whisper推理引擎（单条推理高效优化版）"""

    def __init__(self, model_path: str, config_path: str = "config.yaml", device: str = "cuda"):
        # 算法描述: 初始化模型与处理器，时间复杂度 O(1)，空间复杂度 O(h)
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)

        logger.info(f"加载模型: {model_path}")
        logger.info(f"设备: {self.device}")

        # 加载模型和处理器
        self.processor = WhisperProcessor.from_pretrained(str(self.model_path))
        self.model = WhisperForConditionalGeneration.from_pretrained(str(self.model_path))

        if self.device.type == "cuda":
            self.model.half()  # FP16 推理加速
        self.model.to(self.device)
        self.model.eval()

        logger.info("模型加载完成 ✅")

    def _apply_vocab_constraint(self, text: str, vocab_constraint: set) -> str:
        # 算法描述: 词汇约束过滤，时间复杂度 O(n)，空间复杂度 O(h)
        words = text.lower().split()
        constrained = []
        for w in words:
            cw = "".join(c for c in w if c.isalnum())
            constrained.append(w if w in vocab_constraint or cw in vocab_constraint else "<unk>")
        return " ".join(constrained)

    def transcribe_file(self, audio_path: str, language: str = "en", vocab_constraint: Optional[set] = None) -> Dict[str, any]:
        # 算法描述: 单条推理，时间复杂度 O(n)，空间复杂度 O(h)
        try:
            t_total_start = time.time()
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

            target_sr = getattr(self.processor.feature_extractor, "sampling_rate", 16000)
            if sr != target_sr:
                waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
                sr = target_sr

            audio_np = waveform.cpu().numpy()
            inputs = self.processor(audio_np, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)

            t_infer_start = time.time()

            # 从配置文件读取推理参数
            inference_config = self.config.get("inference", {})
            generate_kwargs = {
                "max_new_tokens": inference_config.get("max_length", 225),
                "num_beams": 1,
                "do_sample": False,
                "early_stopping": True,
                "temperature": inference_config.get("temperature", 0.0),
            }

            # 如果配置中有置信度阈值参数，则添加
            if "logprob_threshold" in inference_config:
                generate_kwargs["logprob_threshold"] = inference_config["logprob_threshold"]
            if "no_speech_threshold" in inference_config:
                generate_kwargs["no_speech_threshold"] = inference_config["no_speech_threshold"]

            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        pred_ids = self.model.generate(input_features, **generate_kwargs)
                else:
                    pred_ids = self.model.generate(input_features, **generate_kwargs)

            t_infer_end = time.time()
            text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

            # 记录生成的token数量用于调试
            num_tokens = pred_ids.shape[1]
            logger.info(f"生成了 {num_tokens} 个tokens，耗时 {t_infer_end - t_infer_start:.3f}s")

            if vocab_constraint:
                text = self._apply_vocab_constraint(text, vocab_constraint)

            duration = len(audio_np) / sr if sr > 0 else 0.0
            infer_time = t_infer_end - t_infer_start
            total_time = time.time() - t_total_start
            rtf = infer_time / duration if duration > 0 else None

            return {
                "text": text,
                "duration": duration,
                "audio_path": audio_path,
                "inference_time": infer_time,
                "total_time": total_time,
                "rtf": rtf
            }

        except Exception as e:
            logger.error(f"转录失败 {audio_path}: {e}")
            return {
                "text": "",
                "duration": 0,
                "audio_path": audio_path,
                "inference_time": 0,
                "total_time": 0,
                "rtf": None,
                "error": str(e)
            }

    def transcribe_batch(self, audio_paths: List[str], language: str = "en") -> List[Dict]:
        # 算法描述: 顺序批量推理，时间复杂度 O(n)，空间复杂度 O(h)
        logger.info(f"批量推理 {len(audio_paths)} 条样本...")
        results = []
        for path in tqdm(audio_paths):
            results.append(self.transcribe_file(path, language))
        return results

    def transcribe_dataset(self, dataset_dir: str, split: str = "test") -> List[Dict]:
        """转录整个数据集"""
        split_dir = Path(dataset_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"找不到 {split_dir}")

        dataset = load_from_disk(str(split_dir))
        results = []

        for item in tqdm(dataset):
            # 修改这里，取正确的字段
            audio_path = item.get("audio_path") or item.get("path")  # 兼容旧数据集
            if audio_path is None:
                continue
            result = self.transcribe_file(audio_path)
            result["reference_text"] = item.get("text", "")
            result["speaker_id"] = item.get("speaker_id", "unknown")
            results.append(result)

        return results

    def transcribe_processed_dataset(self, dataset_dir: str, split: str = "test", language: str = "en") -> List[Dict]:
        """转录预处理后的数据集（直接使用input_features）"""
        split_dir = Path(dataset_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"找不到 {split_dir}")

        dataset = load_from_disk(str(split_dir))
        results = []

        logger.info(f"开始推理 {len(dataset)} 条预处理样本...")

        for idx, item in enumerate(tqdm(dataset)):
            try:
                t_infer_start = time.time()

                # 直接使用预处理好的input_features
                input_features = torch.tensor(item["input_features"], dtype=torch.float32).unsqueeze(0).to(self.device)

                # 从配置文件读取推理参数
                inference_config = self.config.get("inference", {})
                generate_kwargs = {
                    "max_new_tokens": inference_config.get("max_length", 225),
                    "num_beams": 1,
                    "do_sample": False,
                    "early_stopping": True,
                    "temperature": inference_config.get("temperature", 0.0),
                }

                # 如果配置中有置信度阈值参数，则添加
                if "logprob_threshold" in inference_config:
                    generate_kwargs["logprob_threshold"] = inference_config["logprob_threshold"]
                if "no_speech_threshold" in inference_config:
                    generate_kwargs["no_speech_threshold"] = inference_config["no_speech_threshold"]

                with torch.inference_mode():
                    if self.device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            pred_ids = self.model.generate(input_features, **generate_kwargs)
                    else:
                        pred_ids = self.model.generate(input_features, **generate_kwargs)

                t_infer_end = time.time()
                text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

                infer_time = t_infer_end - t_infer_start

                results.append({
                    "text": text,
                    "reference_text": item.get("text", ""),
                    "speaker_id": item.get("speaker_id", "unknown"),
                    "duration": item.get("duration", 0.0),
                    "inference_time": infer_time,
                    "sample_index": idx
                })

            except Exception as e:
                logger.error(f"推理失败 (样本 {idx}): {e}")
                results.append({
                    "text": "",
                    "reference_text": item.get("text", ""),
                    "speaker_id": item.get("speaker_id", "unknown"),
                    "duration": item.get("duration", 0.0),
                    "inference_time": 0.0,
                    "sample_index": idx,
                    "error": str(e)
                })

        return results



class WhisperEvaluator:
    """评估模块：计算WER/CER并保存报告"""

    def __init__(self, config_path: str = "config.yaml"):
        # 算法描述: 初始化评估器，时间复杂度 O(1)，空间复杂度 O(h)
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        logger.info("评估器加载完成 ✅")

    def compute_wer(self, preds: List[str], refs: List[str]) -> float:
        # 算法描述: 计算WER，时间复杂度 O(n)，空间复杂度 O(h)
        return self.wer_metric.compute(predictions=preds, references=refs)

    def compute_cer(self, preds: List[str], refs: List[str]) -> float:
        # 算法描述: 计算CER，时间复杂度 O(n)，空间复杂度 O(h)
        return self.cer_metric.compute(predictions=preds, references=refs)

    def evaluate_results(self, results: List[Dict]) -> Dict[str, float]:
        # 算法描述: 汇总评估结果，时间复杂度 O(n)，空间复杂度 O(h)
        preds = [r["text"] for r in results if "text" in r]
        refs = [r["reference_text"] for r in results if "reference_text" in r]
        wer = self.compute_wer(preds, refs)
        cer = self.compute_cer(preds, refs)

        speaker_wers = {}
        if "speaker_id" in results[0]:
            speakers = {}
            for r in results:
                sid = r.get("speaker_id", "unknown")
                speakers.setdefault(sid, {"pred": [], "ref": []})
                speakers[sid]["pred"].append(r.get("text", ""))
                speakers[sid]["ref"].append(r.get("reference_text", ""))
            for sid, data in speakers.items():
                speaker_wers[sid] = self.compute_wer(data["pred"], data["ref"])

        return {"wer": wer, "cer": cer, "num_samples": len(preds), "speaker_wers": speaker_wers}

    def print_results(self, results: List[Dict], max_samples: int = 10):
        # 算法描述: 打印样本，时间复杂度 O(n)，空间复杂度 O(1)
        logger.info("=" * 100)
        for i, r in enumerate(results[:max_samples]):
            logger.info(f"\n样本 {i+1}")
            logger.info(f"参考: {r.get('reference_text', '')}")
            logger.info(f"预测: {r.get('text', '')}")
            if "speaker_id" in r:
                logger.info(f"说话人: {r['speaker_id']}")
        logger.info("=" * 100)

    def save_results(self, results: List[Dict], output_path: str):
        # 算法描述: 保存结果CSV，时间复杂度 O(n)，空间复杂度 O(h)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"结果已保存到 {path}")

    def generate_report(self, results: List[Dict], metrics: Dict, output_dir: str):
        # 算法描述: 生成报告，时间复杂度 O(n)，空间复杂度 O(h)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {"metrics": metrics, "results": results[:100]}
        with open(out_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.save_results(results, out_dir / "transcription_results.csv")

        logger.info("\n" + "=" * 60)
        logger.info("评估摘要")
        logger.info("=" * 60)
        logger.info(f"样本数: {metrics['num_samples']}")
        logger.info(f"WER: {metrics['wer']:.2%}")
        logger.info(f"CER: {metrics['cer']:.2%}")
        if metrics["speaker_wers"]:
            logger.info("\n按说话人:")
            for sid, wer in sorted(metrics["speaker_wers"].items()):
                logger.info(f"  {sid}: {wer:.2%}")
        logger.info(f"\n报告输出目录: {out_dir}")


def main():
    """主函数 - 演示推理与评估"""
    import argparse
    parser = argparse.ArgumentParser(description="Whisper 高效推理与评估")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--vocab_constraint", type=str)
    parser.add_argument("--use_processed", action="store_true", help="使用预处理后的数据集（input_features）")
    parser.add_argument("--language", type=str, default="en", help="语言代码")
    args = parser.parse_args()

    vocab_constraint = None
    if args.vocab_constraint and Path(args.vocab_constraint).exists():
        with open(args.vocab_constraint, "r", encoding="utf-8") as f:
            vocab_constraint = {line.strip().lower() for line in f if line.strip()}
        logger.info(f"加载词表约束: {len(vocab_constraint)} 词")

    infer = WhisperInference(args.model_path)
    evaluator = WhisperEvaluator()

    if args.audio_path:
        result = infer.transcribe_file(args.audio_path, vocab_constraint=vocab_constraint)
        logger.info(f"\n预测文本: {result['text']}")
        logger.info(f"耗时: {result['inference_time']:.3f}s, RTF={result['rtf']:.3f}")
    elif args.dataset_dir:
        if args.use_processed:
            # 使用预处理后的数据集
            results = infer.transcribe_processed_dataset(args.dataset_dir, args.split, args.language)
        else:
            # 从原始音频推理
            results = infer.transcribe_dataset(args.dataset_dir, args.split)
        metrics = evaluator.evaluate_results(results)
        evaluator.print_results(results)
        evaluator.generate_report(results, metrics, args.output_dir)


if __name__ == "__main__":
    main()
