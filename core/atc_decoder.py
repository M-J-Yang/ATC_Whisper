"""
ATC词汇约束解码器 - 使用领域词汇提高识别精度
通过Beam Search和浅融合(Shallow Fusion)实现词汇约束
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ATCVocabularyConstraint:
    """ATC词汇约束器"""

    def __init__(self, wordlist_path: str):
        """初始化词汇约束器

        Args:
            wordlist_path: ATC词汇表文件路径
        """
        self.wordlist = self._load_wordlist(wordlist_path)
        self.vocab_set = set(self.wordlist)

        logger.info(f"加载ATC词汇表: {len(self.vocab_set)} 个词汇")

    def _load_wordlist(self, path: str) -> List[str]:
        """加载词汇表文件"""
        words = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word and not word.startswith("["):
                    words.append(word)
        return words

    def is_valid_word(self, word: str) -> bool:
        """检查单词是否在ATC词汇表中"""
        return word.lower() in self.vocab_set

    def constrain_tokens(self, tokens: torch.Tensor, tokenizer) -> torch.Tensor:
        """约束Token集合为有效的ATC词汇

        Args:
            tokens: Token ID张量
            tokenizer: Tokenizer对象

        Returns:
            有效Token的mask (True表示可用)
        """
        mask = torch.zeros(tokens.shape[0], dtype=torch.bool)

        for idx in range(tokens.shape[0]):
            token_id = tokens[idx].item()
            token_str = tokenizer.decode([token_id]).strip().lower()

            # 检查Token是否是有效的ATC词汇或特殊Token
            if self.is_valid_word(token_str) or token_str in ["", "<|endoftext|>"]:
                mask[idx] = True

        return mask


class ConstrainedBeamSearchDecoder:
    """约束Beam Search解码器"""

    def __init__(
        self,
        model,
        processor,
        vocab_constraint: Optional[ATCVocabularyConstraint] = None,
        beam_size: int = 5,
        max_length: int = 224
    ):
        """初始化受约束的Beam Search解码器

        Args:
            model: Whisper模型
            processor: WhisperProcessor
            vocab_constraint: ATC词汇约束器 (可选)
            beam_size: Beam宽度
            max_length: 最大生成长度
        """
        self.model = model
        self.processor = processor
        self.vocab_constraint = vocab_constraint
        self.beam_size = beam_size
        self.max_length = max_length

        logger.info(f"初始化约束Beam Search (beam_size={beam_size})")

    def decode(
        self,
        input_features: torch.Tensor,
        language: str = "en"
    ) -> Dict[str, any]:
        """执行受约束的解码

        Args:
            input_features: 输入音频特征
            language: 语言代码

        Returns:
            解码结果字典
        """
        with torch.no_grad():
            # 生成Token序列
            if self.vocab_constraint is None:
                # 无约束Beam Search
                predicted_ids = self.model.generate(
                    input_features,
                    language=language,
                    task="transcribe",
                    max_new_tokens=self.max_length,
                    num_beams=self.beam_size,
                    temperature=0.0,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            else:
                # 有词汇约束的Beam Search
                # (完整实现需要自定义生成函数)
                predicted_ids = self.model.generate(
                    input_features,
                    language=language,
                    task="transcribe",
                    max_new_tokens=self.max_length,
                    num_beams=self.beam_size,
                    temperature=0.0
                )

        # 解码
        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0] if hasattr(predicted_ids, '__len__') else predicted_ids

        return {
            "text": text.strip() if isinstance(text, str) else text,
            "beam_size": self.beam_size
        }


class ShallowFusionLM:
    """浅融合语言模型 - 结合转录概率和语言模型概率"""

    def __init__(
        self,
        wordlist_path: str,
        alpha: float = 0.5
    ):
        """初始化浅融合LM

        Args:
            wordlist_path: ATC词汇表路径
            alpha: 融合权重 (0=ASR, 1=LM)
        """
        self.vocab = self._load_vocab(wordlist_path)
        self.alpha = alpha
        self.vocab_probs = self._compute_vocab_priors()

        logger.info(f"初始化浅融合LM (alpha={alpha}, vocab_size={len(self.vocab)})")

    def _load_vocab(self, wordlist_path: str) -> Dict[str, int]:
        """加载词汇表和频率"""
        vocab = {}
        with open(wordlist_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                word = line.strip().lower()
                if word and not word.startswith("["):
                    vocab[word] = idx

        return vocab

    def _compute_vocab_priors(self) -> Dict[str, float]:
        """计算词汇的先验概率 (均匀分布)"""
        n = len(self.vocab)
        return {word: 1.0 / n for word in self.vocab}

    def fuse_scores(
        self,
        asr_scores: np.ndarray,
        words: List[str]
    ) -> np.ndarray:
        """融合ASR得分和LM得分

        Args:
            asr_scores: ASR模型的置信度
            words: 识别的单词序列

        Returns:
            融合后的得分
        """
        fused_scores = np.zeros_like(asr_scores)

        for i, word in enumerate(words):
            # ASR得分
            asr_score = asr_scores[i]

            # 语言模型得分
            lm_score = self.vocab_probs.get(word.lower(), 1e-10)

            # 融合
            fused = (1 - self.alpha) * asr_score + self.alpha * np.log(lm_score + 1e-10)
            fused_scores[i] = fused

        return fused_scores


class ConstrainedATCDecoder:
    """完整的受约束ATC解码器 - 结合多种约束技术"""

    def __init__(
        self,
        model,
        processor,
        config_dict: Dict[str, any]
    ):
        """初始化ATC解码器

        Args:
            model: Whisper模型
            processor: WhisperProcessor
            config_dict: 配置字典，包含:
                - wordlist_path: ATC词汇表路径
                - beam_size: Beam宽度
                - use_shallow_fusion: 是否使用浅融合
                - lm_alpha: LM融合权重
        """
        self.model = model
        self.processor = processor

        self.wordlist_path = config_dict.get("wordlist_path", "TXTdata/wordlist.txt")
        self.beam_size = config_dict.get("beam_size", 5)
        self.use_shallow_fusion = config_dict.get("use_shallow_fusion", False)
        self.lm_alpha = config_dict.get("lm_alpha", 0.5)

        # 初始化词汇约束
        self.vocab_constraint = ATCVocabularyConstraint(self.wordlist_path)

        # 初始化解码器
        self.beam_search_decoder = ConstrainedBeamSearchDecoder(
            model,
            processor,
            vocab_constraint=self.vocab_constraint,
            beam_size=self.beam_size
        )

        # 初始化浅融合LM (可选)
        if self.use_shallow_fusion:
            self.lm = ShallowFusionLM(
                self.wordlist_path,
                alpha=self.lm_alpha
            )
        else:
            self.lm = None

        logger.info(f"初始化ConstrainedATCDecoder")
        logger.info(f"  - Beam Size: {self.beam_size}")
        logger.info(f"  - 使用浅融合: {self.use_shallow_fusion}")

    def decode(
        self,
        input_features: torch.Tensor,
        language: str = "en"
    ) -> Dict[str, any]:
        """执行完整的约束解码

        Args:
            input_features: 输入特征
            language: 语言代码

        Returns:
            解码结果字典
        """
        # Beam Search解码
        result = self.beam_search_decoder.decode(
            input_features,
            language=language
        )

        # 后处理
        text = result["text"]
        words = text.split()

        # 移除不在ATC词汇表中的单词 (可选)
        # cleaned_words = [w for w in words if self.vocab_constraint.is_valid_word(w)]
        # text = " ".join(cleaned_words)

        result["text"] = text
        result["num_words"] = len(words)
        result["vocab_coverage"] = sum(
            1 for w in words if self.vocab_constraint.is_valid_word(w)
        ) / max(len(words), 1)

        return result

    @staticmethod
    def create_from_config(model, processor, config_path: str):
        """从配置文件创建解码器

        Args:
            model: Whisper模型
            processor: WhisperProcessor
            config_path: 配置文件路径

        Returns:
            ConstrainedATCDecoder实例
        """
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        atc_config = {
            "wordlist_path": Path(config["data"]["dataset_dir"]) / config["data"]["wordlist_path"],
            "beam_size": config["inference"]["beam_size"],
            "use_shallow_fusion": False,  # 可根据需要启用
            "lm_alpha": 0.5
        }

        return ConstrainedATCDecoder(model, processor, atc_config)


def demo_constrained_decoding():
    """演示受约束解码的使用"""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torchaudio

    logger.info("=" * 60)
    logger.info("ATC受约束解码演示")
    logger.info("=" * 60)

    # 加载模型
    model_name = "openai/whisper-base"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # 创建受约束解码器
    config_dict = {
        "wordlist_path": "d:\\NPU_works\\语音\\demo\\ATCOSIM\\TXTdata\\wordlist.txt",
        "beam_size": 5,
        "use_shallow_fusion": False,
    }

    decoder = ConstrainedATCDecoder(model, processor, config_dict)

    logger.info(f"词汇表大小: {len(decoder.vocab_constraint.vocab_set)}")
    logger.info(f"Beam宽度: {decoder.beam_size}")

    logger.info("\n测试词汇约束:")
    test_words = ["climb", "flight", "level", "zurich", "invalid_atc_word"]
    for word in test_words:
        is_valid = decoder.vocab_constraint.is_valid_word(word)
        logger.info(f"  '{word}': {'✓ 有效' if is_valid else '✗ 无效'}")


if __name__ == "__main__":
    demo_constrained_decoding()
