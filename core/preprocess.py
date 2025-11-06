"""
数据预处理管道 - ATCOSIM语音识别数据准备
处理流程: CSV解析 -> 音频重采样 -> 数据集划分 -> 保存为标准格式
"""

import os
import json
import csv
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import yaml
from datasets import Dataset, DatasetDict
from transformers import WhisperProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATCOSIMProcessor:
    """ATCOSIM数据集处理器"""

    def __init__(self, config_path: str = "config.yaml"):
        """初始化处理器"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.dataset_dir = Path(self.config["data"]["dataset_dir"])
        self.csv_path = self.dataset_dir / self.config["data"]["csv_path"]
        self.output_dir = Path(self.config["output"]["output_dir"]) / "processed_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate = self.config["data"]["target_sr"]

        # 加载 Whisper Processor 用于特征提取
        model_type = self.config["model"]["type"]
        if model_type == "whisper":
            whisper_size = self.config["model"]["whisper_size"]
            model_name = f"openai/whisper-{whisper_size}"
            logger.info(f"加载 Whisper Processor: {model_name}")
            self.processor = WhisperProcessor.from_pretrained(model_name)
        else:
            self.processor = None

        logger.info(f"数据集目录: {self.dataset_dir}")
        logger.info(f"输出目录: {self.output_dir}")

    def load_csv_metadata(self) -> pd.DataFrame:
        """加载CSV元数据"""
        logger.info("正在加载 fulldata.csv...")
        df = pd.read_csv(self.csv_path)
        logger.info(f"加载完成: {len(df)} 条记录")

        # 过滤损坏的音频
        df = df[df['recording_corrupt'] == 0]
        logger.info(f"过滤后: {len(df)} 条记录 (已移除损坏的音频)")

        return df

    def load_wordlist(self) -> set:
        """加载ATC词汇表"""
        wordlist_path = self.dataset_dir / self.config["data"]["wordlist_path"]
        with open(wordlist_path, "r", encoding="utf-8") as f:
            words = {line.strip().lower() for line in f if line.strip()}
        logger.info(f"加载词汇表: {len(words)} 个单词")
        return words

    def normalize_transcription(self, text: str) -> str:
        """标准化转录文本 - ATC特化版本
        - 移除/映射特殊标记 (~p=pause, ~s=unclear, ~a=accent)
        - 转小写
        - 保留数字和常用ATC术语
        - 移除多余空格
        """
        # 移除ATC特殊标记但保留含义
        # ~p 表示停顿，~s 表示不清楚，~a 表示口音
        text = text.replace("~p", "").replace("~s", "").replace("~a", "")

        # 小写并移除leading/trailing空格
        text = text.lower().strip()

        # 移除多余空格（保留单个空格分隔单词）
        text = " ".join(text.split())

        # 保留字母数字和空格，移除特殊字符（除了对ATC有意义的）
        text = "".join(c for c in text if c.isalnum() or c.isspace())

        return text

    def load_and_resample_audio(self, wav_path: Path) -> Tuple[np.ndarray, int]:
        """加载并重采样音频"""
        try:
            audio, sr = librosa.load(str(wav_path), sr=self.sample_rate, mono=True)

            if self.config["data"]["normalize"]:
                # 音量归一化
                audio = audio / (np.max(np.abs(audio)) + 1e-8)

            return audio, self.sample_rate
        except Exception as e:
            logger.warning(f"加载失败 {wav_path}: {e}")
            return None, None

    def apply_speed_perturbation(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """应用速度扰动（Speed Perturbation）
        Args:
            audio: 音频数组
            sr: 采样率
        Returns:
            增强后的音频
        """
        if not self.config["data"]["augmentation"]["enabled"]:
            return audio

        # 随机选择速度因子
        speed_factors = self.config["data"]["augmentation"]["speed_perturb"]
        if speed_factors and len(speed_factors) >= 2:
            speed_factor = np.random.uniform(speed_factors[0], speed_factors[1])
        else:
            return audio

        # 使用librosa进行速度调整
        try:
            augmented_audio = librosa.effects.time_stretch(audio, rate=speed_factor)
            return augmented_audio
        except Exception as e:
            logger.warning(f"速度扰动失败: {e}，返回原始音频")
            return audio

    def apply_spec_augment(self, audio: np.ndarray, sr: int,
                          freq_mask_param: int = 30, time_mask_param: int = 40) -> np.ndarray:
        """应用SpecAugment增强（时频掩码）

        Args:
            audio: 音频数组
            sr: 采样率
            freq_mask_param: 频率掩码宽度
            time_mask_param: 时间掩码宽度
        Returns:
            增强后的音频
        """
        if not self.config["data"]["augmentation"]["enabled"]:
            return audio

        try:
            # 计算MFCC特征
            S = librosa.feature.melspectrogram(y=audio, sr=sr)

            # 频率掩码
            freq_mask_width = np.random.randint(0, freq_mask_param)
            if freq_mask_width > 0:
                freq_mask_start = np.random.randint(0, S.shape[0] - freq_mask_width)
                S[freq_mask_start:freq_mask_start + freq_mask_width, :] = 0

            # 时间掩码
            time_mask_width = np.random.randint(0, time_mask_param)
            if time_mask_width > 0:
                time_mask_start = np.random.randint(0, S.shape[1] - time_mask_width)
                S[:, time_mask_start:time_mask_start + time_mask_width] = 0

            # 从修改后的频谱转换回音频（近似）
            # 注：这是一个简化版本，实际SpecAugment通常在特征层面应用
            return audio  # 返回原始音频（SpecAugment在特征提取阶段更有效）
        except Exception as e:
            logger.warning(f"SpecAugment失败: {e}，返回原始音频")
            return audio

    def build_dataset(self, df: pd.DataFrame) -> List[Dict]:
        """构建数据集

        返回格式:
        [
            {
                "audio_path": "path/to/audio.wav",
                "text": "transcription",
                "speaker_id": "sm1",
                "session_id": "sm1_01",
                "utterance_id": 1,
                "duration": 3.3
            },
            ...
        ]
        """
        dataset = []
        failed_count = 0

        logger.info("正在构建数据集...")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # 构造音频文件路径
                directory = row['directory']
                subdirectory = row['subdirectory']
                filename = row['filename']

                wav_path = self.dataset_dir / "WAVdata" / directory / subdirectory / f"{filename}.wav"

                if not wav_path.exists():
                    logger.warning(f"文件不存在: {wav_path}")
                    failed_count += 1
                    continue

                # 加载并重采样音频
                audio, sr = self.load_and_resample_audio(wav_path)
                if audio is None:
                    failed_count += 1
                    continue

                # 标准化转录
                text = self.normalize_transcription(row['transcription'])

                # 跳过空转录
                if not text or len(text) < 2:
                    logger.warning(f"空转录: {filename}")
                    failed_count += 1
                    continue

                # 创建数据项
                data_item = {
                    "audio_path": str(wav_path),
                    "audio_data": audio,  # 临时保存原始数据
                    "text": text,
                    "speaker_id": str(row['speaker_id']),
                    "session_id": str(row['session_id']),
                    "utterance_id": int(row['utterance_id']),
                    "duration": float(row['length_sec']),
                    "recording_id": str(row['recording_id'])
                }

                dataset.append(data_item)

            except Exception as e:
                logger.error(f"处理失败 (行 {idx}): {e}")
                failed_count += 1
                continue

        logger.info(f"构建完成: {len(dataset)} 条有效记录, {failed_count} 条失败")
        return dataset

    def split_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """数据集划分 - 按说话人分层"""
        speakers = {}
        for item in dataset:
            speaker = item["speaker_id"]
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(item)

        train_data, val_data, test_data = [], [], []

        for speaker, items in speakers.items():
            n_speaker = len(items)
            train_end = int(n_speaker * self.config["data"]["train_split"])
            val_end = train_end + int(n_speaker * self.config["data"]["val_split"])

            train_data.extend(items[:train_end])
            val_data.extend(items[train_end:val_end])
            test_data.extend(items[val_end:])

        logger.info(f"数据集划分: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        return train_data, val_data, test_data

    def save_processed_data_huggingface(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """保存为 HuggingFace Datasets 格式（最快训练速度）

        预先提取 mel spectrogram 特征和 tokenize 文本，训练时零开销
        采用分批处理策略避免 OOM
        """
        logger.info("正在创建 HuggingFace Datasets（预提取特征）...")

        def create_dataset_batched(data: List[Dict], split_name: str, batch_size: int = 500) -> Dataset:
            """分批提取特征并创建 Dataset，避免内存溢出"""
            logger.info(f"处理 {split_name} 数据集 ({len(data)} 条)，批次大小={batch_size}")

            all_datasets = []

            # 分批处理
            for batch_idx in range(0, len(data), batch_size):
                batch_data = data[batch_idx:batch_idx + batch_size]
                batch_num = batch_idx // batch_size + 1
                total_batches = (len(data) + batch_size - 1) // batch_size

                logger.info(f"处理批次 {batch_num}/{total_batches} ({len(batch_data)} 条)...")

                # 准备数据字典
                dataset_dict = {
                    "input_features": [],
                    "labels": [],
                    "text": [],
                    "speaker_id": [],
                    "duration": []
                }

                for item in tqdm(batch_data, desc=f"批次 {batch_num}/{total_batches}"):
                    try:
                        # 提取音频的 mel spectrogram 特征
                        audio_array = item["audio_data"]
                        inputs = self.processor.feature_extractor(
                            audio_array,
                            sampling_rate=self.sample_rate,
                            return_tensors="np"
                        )

                        # Tokenize 文本标签
                        labels = self.processor.tokenizer(
                            item["text"],
                            return_tensors="np"
                        )

                        # 存储特征（转为 list 以便序列化）
                        dataset_dict["input_features"].append(inputs.input_features[0].tolist())
                        dataset_dict["labels"].append(labels.input_ids[0].tolist())
                        dataset_dict["text"].append(item["text"])
                        dataset_dict["speaker_id"].append(item["speaker_id"])
                        dataset_dict["duration"].append(item["duration"])

                    except Exception as e:
                        logger.warning(f"处理失败: {item.get('recording_id', 'unknown')}, 错误: {e}")
                        continue

                # 创建当前批次的 Dataset
                batch_dataset = Dataset.from_dict(dataset_dict)
                all_datasets.append(batch_dataset)

                # 释放内存
                del dataset_dict, batch_data
                import gc
                gc.collect()

            # 合并所有批次
            logger.info(f"合并 {len(all_datasets)} 个批次...")
            from datasets import concatenate_datasets
            final_dataset = concatenate_datasets(all_datasets)

            # 释放内存
            del all_datasets
            import gc
            gc.collect()

            return final_dataset

        # 分批创建三个子数据集
        # 根据系统内存调整 batch_size：
        # - 90GB+ 内存: batch_size=2000 (约 4 批次，快速)
        # - 32GB 内存: batch_size=1000 (约 8 批次，平衡)
        # - 16GB 内存: batch_size=500 (约 16 批次，安全)
        logger.info("=" * 60)
        train_dataset = create_dataset_batched(train_data, "train", batch_size=2000)
        del train_data
        import gc
        gc.collect()

        logger.info("=" * 60)
        val_dataset = create_dataset_batched(val_data, "val", batch_size=1000)
        del val_data
        gc.collect()

        logger.info("=" * 60)
        test_dataset = create_dataset_batched(test_data, "test", batch_size=1000)
        del test_data
        gc.collect()

        # 组合成 DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        })

        # 保存到磁盘
        logger.info("=" * 60)
        logger.info(f"保存数据集到 {self.output_dir}...")
        dataset_dict.save_to_disk(str(self.output_dir))

        logger.info(f"✅ 数据集已保存为 HuggingFace 格式（特征已预提取）")
        logger.info(f"   - train: {len(train_dataset)} 条")
        logger.info(f"   - val: {len(val_dataset)} 条")
        logger.info(f"   - test: {len(test_dataset)} 条")
        logger.info(f"   - 路径: {self.output_dir}")

    def process(self):
        """执行完整的数据处理流程"""
        logger.info("=" * 60)
        logger.info("开始ATCOSIM数据处理 (HuggingFace Datasets 格式)")
        logger.info("=" * 60)

        # 1. 加载元数据
        df = self.load_csv_metadata()

        # 2. 加载词汇表
        wordlist = self.load_wordlist()

        # 3. 构建数据集
        dataset = self.build_dataset(df)

        # 4. 划分数据集
        train_data, val_data, test_data = self.split_dataset(dataset)

        # 5. 保存为 HuggingFace Datasets 格式（最优性能）
        self.save_processed_data_huggingface(train_data, val_data, test_data)

        logger.info("=" * 60)
        logger.info("✅ 数据处理完成！")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)

        return train_data, val_data, test_data


def main():
    """主函数"""
    processor = ATCOSIMProcessor(config_path="config.yaml")
    train_data, val_data, test_data = processor.process()


if __name__ == "__main__":
    main()
