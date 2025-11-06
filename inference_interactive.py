"""
交互式推理脚本
加载模型一次后，可以连续推理多个音频文件
"""

import logging
from pathlib import Path
from inference_service import get_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """交互式推理主函数"""

    # 获取推理服务（只加载一次）
    service = get_service()
    config = service.get_config()

    logger.info("\n" + "=" * 60)
    logger.info("交互式推理模式")
    logger.info("=" * 60)
    logger.info("输入音频文件路径进行推理")
    logger.info("输入 'quit' 或 'exit' 退出")
    logger.info("输入 'config' 显示当前配置")
    logger.info("=" * 60)

    # 是否使用词汇约束
    use_vocab = config.get("single_wav", {}).get("use_vocab_constraint", False)
    logger.info(f"词汇约束: {'启用' if use_vocab else '禁用'}")
    logger.info("=" * 60 + "\n")

    # 交互式循环
    while True:
        try:
            # 获取用户输入
            audio_path = input("请输入音频文件路径（或命令）: ").strip()

            # 处理退出命令
            if audio_path.lower() in ['quit', 'exit', 'q']:
                logger.info("退出交互式推理模式")
                break

            # 显示配置
            if audio_path.lower() == 'config':
                logger.info("\n当前配置:")
                logger.info(f"  模型路径: {service.model_path}")
                logger.info(f"  语言: {service.language}")
                logger.info(f"  设备: {service.device}")
                logger.info(f"  词汇约束: {'启用' if use_vocab else '禁用'}")
                if service.vocab_constraint:
                    logger.info(f"  词汇数量: {len(service.vocab_constraint)}")
                print()
                continue

            # 跳过空输入
            if not audio_path:
                continue

            # 检查文件是否存在
            if not Path(audio_path).exists():
                logger.error(f"文件不存在: {audio_path}\n")
                continue

            # 执行推理
            logger.info(f"\n推理中: {audio_path}")
            result = service.transcribe(audio_path, use_vocab_constraint=use_vocab)

            # 检查错误
            if "error" in result:
                logger.error(f"推理失败: {result['error']}\n")
                continue

            # 显示结果
            logger.info("-" * 60)
            logger.info(f"识别文本: {result['text']}")
            logger.info(f"音频时长: {result['duration']:.2f}秒")
            logger.info(f"推理耗时: {result['inference_time']:.3f}秒")
            if result.get('rtf'):
                logger.info(f"实时率(RTF): {result['rtf']:.3f}")
            logger.info("-" * 60 + "\n")

        except KeyboardInterrupt:
            logger.info("\n\n检测到 Ctrl+C，退出程序")
            break
        except Exception as e:
            logger.error(f"错误: {e}\n")
            continue


if __name__ == "__main__":
    main()
