"""
单条语音推理脚本
从config.yaml读取配置，对单条音频进行语音识别
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import logging
from backend.inference_service import get_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数 - 单条语音推理"""

    # 获取推理服务（会自动加载模型和预热）
    service = get_service()
    config = service.get_config()

    # 获取单条语音路径
    audio_path = config.get("single_wav", {}).get("path")
    if not audio_path:
        logger.error("错误: config.yaml中未配置single_wav.path")
        logger.info("请在config.yaml中添加:")
        logger.info("single_wav:")
        logger.info("  path: \"your_audio_file.wav\"")
        return

    # 检查文件是否存在
    if not Path(audio_path).exists():
        logger.error(f"错误: 音频文件不存在: {audio_path}")
        return

    # 执行推理
    logger.info("\n" + "=" * 60)
    logger.info("开始推理")
    logger.info("=" * 60)
    logger.info(f"音频文件: {audio_path}")

    try:
        # 是否使用词汇约束
        use_vocab = config.get("single_wav", {}).get("use_vocab_constraint", False)

        # 执行推理
        result = service.transcribe(audio_path, use_vocab_constraint=use_vocab)

        # 检查是否有错误
        if "error" in result:
            logger.error(f"推理失败: {result['error']}")
            return

        # 输出结果
        logger.info("=" * 60)
        logger.info("推理结果")
        logger.info("=" * 60)
        logger.info(f"识别文本: {result['text']}")
        logger.info(f"音频时长: {result['duration']:.2f}秒")
        logger.info(f"推理耗时: {result['inference_time']:.3f}秒")
        logger.info(f"总耗时: {result['total_time']:.3f}秒")
        if result.get('rtf'):
            logger.info(f"实时率(RTF): {result['rtf']:.3f}")
        logger.info("=" * 60)

        # 保存结果到文件（可选）
        save_result = config.get("single_wav", {}).get("save_result", False)
        if save_result:
            output_file = config.get("single_wav", {}).get("output_file", "single_inference_result.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"音频文件: {audio_path}\n")
                f.write(f"识别文本: {result['text']}\n")
                f.write(f"音频时长: {result['duration']:.2f}秒\n")
                f.write(f"推理耗时: {result['inference_time']:.3f}秒\n")
            logger.info(f"\n结果已保存到: {output_file}")

    except Exception as e:
        logger.error(f"推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
