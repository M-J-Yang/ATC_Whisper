"""
推理服务模块
负责模型加载、预热和管理
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import torch
import logging
from core.inference import WhisperInference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceService:
    """推理服务类 - 单例模式管理模型"""

    _instance = None
    _initialized = False

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = None):
        # 避免重复初始化
        if self._initialized:
            return

        # 默认配置路径为项目根目录的 config.yaml
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        logger.info("=" * 60)
        logger.info("初始化推理服务")
        logger.info("=" * 60)

        # 加载配置
        self.config = self._load_config(config_path)

        # 获取模型路径
        self.model_path = self.config.get("single_wav", {}).get("model_path")
        if not self.model_path:
            whisper_size = self.config.get("model", {}).get("whisper_size", "base")
            self.model_path = f"openai/whisper-{whisper_size}"

        logger.info(f"模型路径: {self.model_path}")

        # 获取语言配置
        self.language = self.config.get("inference", {}).get("language", "en")

        # 加载词汇约束
        self.vocab_constraint = self._load_vocab_constraint()

        # 检查GPU
        self.device = self._check_device()

        # 加载模型
        logger.info("正在加载模型...")
        self.infer = WhisperInference(self.model_path, config_path=str(config_path), device=self.device)
        logger.info("模型加载完成 ✅")

        # GPU预热
        if self.device == "cuda":
            self._warmup()

        self._initialized = True
        logger.info("=" * 60)
        logger.info("推理服务初始化完成！")
        logger.info("=" * 60)

    def _load_config(self, config_path) -> dict:
        """加载配置文件"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config

    def _load_vocab_constraint(self):
        """加载词汇约束"""
        use_vocab_constraint = self.config.get("single_wav", {}).get("use_vocab_constraint", False)
        if not use_vocab_constraint:
            logger.info("词汇约束: 禁用")
            return None

        dataset_dir = self.config.get("data", {}).get("dataset_dir", "ATCOSIM/")
        wordlist_relative = self.config.get("data", {}).get("wordlist_path", "TXTdata/wordlist.txt")

        # 相对于项目根目录
        project_root = Path(__file__).parent.parent
        if wordlist_relative.startswith(dataset_dir):
            vocab_path = project_root / wordlist_relative
        else:
            vocab_path = project_root / dataset_dir / wordlist_relative

        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_constraint = {line.strip().lower() for line in f if line.strip()}
            logger.info(f"词汇约束: 启用 ({len(vocab_constraint)} 词)")
            return vocab_constraint
        else:
            logger.warning(f"词表文件不存在: {vocab_path}，词汇约束已禁用")
            return None

    def _check_device(self):
        """检查并配置设备"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"检测到GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"PyTorch版本: {torch.__version__}")
        else:
            device = "cpu"
            logger.warning("未检测到可用GPU，将使用CPU运行（会很慢）")
        return device

    def _warmup(self):
        """GPU预热"""
        logger.info("正在预热GPU...")
        # 创建一个虚拟的短音频进行预热
        import numpy as np
        import tempfile
        import torchaudio

        dummy_audio = np.random.randn(16000).astype(np.float32)  # 1秒音频

        # 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            torchaudio.save(tmp_path, torch.from_numpy(dummy_audio).unsqueeze(0), 16000)

        # 预热推理
        _ = self.infer.transcribe_file(tmp_path, language=self.language, vocab_constraint=None)

        # 删除临时文件
        Path(tmp_path).unlink()
        logger.info("GPU预热完成 ✅")

    def transcribe(self, audio_path: str, use_vocab_constraint: bool = True):
        """
        执行推理

        Args:
            audio_path: 音频文件路径
            use_vocab_constraint: 是否使用词汇约束

        Returns:
            dict: 推理结果
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        vocab = self.vocab_constraint if use_vocab_constraint else None

        result = self.infer.transcribe_file(
            audio_path,
            language=self.language,
            vocab_constraint=vocab
        )

        return result

    def get_config(self):
        """获取配置"""
        return self.config


# 全局服务实例
_service = None


def get_service(config_path: str = None) -> InferenceService:
    """获取推理服务实例（单例）"""
    global _service
    if _service is None:
        _service = InferenceService(config_path)
    return _service
