"""
构建高质量ATC词汇库 - 单词级别，不是句子级别
使用jieba分词 + 词性过滤 + 手动规范词汇
"""

import jieba
import jieba.posseg as pseg
from pathlib import Path
from collections import Counter
import json
import re

# 手动定义的ATC标准词汇 - 这是基础词汇库
ATC_STANDARD_VOCAB = {
    # 天气相关
    'weather': [
        '晴朗', '晴', '多云', '少云', '阴', '雨', '雪', '雷', '闪电', '露点',
        '能见度', '风', '温度', '压力', '场压', '气压', '天气', '无限',
        '厚', '薄', '弱', '强', '冷', '热', '干', '湿', '低气压', '高气压',
        '逆温', '风切变', '微风', '强风', '阵风', '下沉气流', '上升气流'
    ],

    # 飞行操作
    'flight_ops': [
        '起飞', '着陆', '降落', '滑行', '爬升', '下降', '巡航', '转弯',
        '改出', '拉平', '加速', '减速', '进近', '复飞', '脱离', '进跑道',
        '滑出', '报告', '确认', '收到', '明白', '呼叫', '联系', '完毕',
        '等待', '保持', '高度', '速度', '航向', '距离', '方向', '五边',
        '四边', '三转弯', '二转弯', '左转', '右转', '直飞', '绕行'
    ],

    # 飞机部件
    'aircraft_parts': [
        '油箱', '副油箱', '发动机', '起落架', '襟翼', '方向舵', '升降舵',
        '副翼', '轮子', '轮胎', '燃油', '液压', '电气', '通信', '导航',
        '雷达', '自动驾驶', '防冰', '防火', '舱门', '应答机', '高度表',
        '速度表', '罗盘', '玻璃'
    ],

    # 方向和度数
    'directions': [
        '左', '右', '前', '后', '北', '南', '东', '西', '度', '向',
        '东北', '西北', '东南', '西南', '顺风', '逆风', '侧风'
    ],

    # 数字相关
    'numbers': [
        '零', '一', '二', '三', '四', '五', '六', '七', '八', '九',
        '十', '百', '千', '万', '点', '米', '秒', '分', '时', '公里'
    ],

    # 通话用语
    'communication': [
        '请', '请求', '允许', '批准', '否决', '禁止', '注意', '警告',
        '重复', '再说一遍', '说慢点', '大声点', '清楚', '不清楚', '干扰',
        '信号', '中断', '恢复', '再见', '祝安全', '谢谢'
    ],

    # 机场和地标
    'airports': [
        '泰山', '长江', '锦州', '跑道', '停机坪', '跑道口', '跑道入侵',
        '盘旋', 'P256', 'P283', 'P377', 'P322'
    ],

    # 其他重要词汇
    'misc': [
        '位置', '报告', '通过', '进入', '离开', '区域', '航线', '放好',
        '检查', '工作', '状态', '情况', '好', '可以', '不可以', '收到'
    ]
}

def extract_sentences_from_dataset(data_dir="../chinese_ATC_formatted/TXTdata"):
    """从数据集提取所有句子"""

    sentences = []
    txt_dir = Path(data_dir)

    if not txt_dir.exists():
        print(f"❌ 数据目录不存在: {txt_dir}")
        return []

    txt_files = list(txt_dir.glob("**/*.txt"))
    print(f"📖 处理 {len(txt_files)} 个文本文件...")

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # 每行是一个句子
                for line in content.split('\n'):
                    if line.strip():
                        sentences.append(line.strip())
        except Exception as e:
            print(f"⚠️  错误读取 {txt_file}: {e}")

    return sentences

def segment_and_filter_words(sentences):
    """使用jieba分词，提取真正的词汇"""

    word_counter = Counter()
    char_set = set()

    print(f"🔪 分词 {len(sentences)} 条句子...")

    for i, sentence in enumerate(sentences):
        if i % 100 == 0:
            print(f"   处理进度: {i}/{len(sentences)}")

        try:
            # 使用jieba进行词性标注
            words_pos = pseg.cut(sentence)

            for word, pos in words_pos:
                # 过滤条件：
                # 1. 词长2-6个字
                # 2. 词性是名词(n)、动词(v)、形容词(a)、数词(m)、时间词(t)
                # 3. 不是标点或其他
                if 2 <= len(word) <= 6 and pos in ['n', 'v', 'a', 'm', 't', 'ad', 'an', 'vn']:
                    word_counter[word] += 1
                    for char in word:
                        char_set.add(char)

        except Exception as e:
            print(f"⚠️  分词错误: {e}")

    return word_counter, char_set

def build_quality_vocab(word_counter, min_frequency=2):
    """构建高质量的词汇库"""

    # 过滤低频词
    filtered_words = {word: count for word, count in word_counter.items() if count >= min_frequency}

    print(f"✅ 提取词汇: {len(filtered_words)} 个")
    print(f"   最低频率阈值: {min_frequency}")

    return filtered_words

def merge_with_standard_vocab(extracted_words, standard_vocab):
    """合并自动提取和标准词汇"""

    # 展平标准词汇
    standard_words = set()
    for category, words in standard_vocab.items():
        standard_words.update(words)

    # 合并
    merged = set(extracted_words.keys())
    merged.update(standard_words)

    print(f"\n📊 词汇合并统计:")
    print(f"   自动提取: {len(extracted_words)}")
    print(f"   标准词汇: {len(standard_words)}")
    print(f"   合并后: {len(merged)}")

    return merged, standard_words

def save_vocab_resources(vocab_set, standard_vocab, extracted_words, output_dir='./atc_vocab'):
    """保存词汇资源"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. 保存分类词汇表
    with open(output_dir / 'atc_vocab_classified.json', 'w', encoding='utf-8') as f:
        json.dump(standard_vocab, f, ensure_ascii=False, indent=2)

    # 2. 保存完整词汇表（推理约束用）
    with open(output_dir / 'atc_vocab.txt', 'w', encoding='utf-8') as f:
        for word in sorted(vocab_set):
            f.write(word + '\n')

    # 3. 保存词频统计（只有自动提取的）
    stats = {}
    for category, words in standard_vocab.items():
        stats[category] = {
            'count': len(words),
            'words': words
        }

    with open(output_dir / 'vocab_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 4. 保存自动提取的词汇频率
    top_extracted = sorted(extracted_words.items(), key=lambda x: x[1], reverse=True)[:100]
    with open(output_dir / 'extracted_vocab_freq.json', 'w', encoding='utf-8') as f:
        json.dump(dict(top_extracted), f, ensure_ascii=False, indent=2)

    print(f"\n✅ 词汇资源已保存到 {output_dir}:")
    print(f"   📄 atc_vocab.txt - 完整词表 ({len(vocab_set)} 词)")
    print(f"   📄 atc_vocab_classified.json - 分类词汇")
    print(f"   📄 vocab_stats.json - 词汇统计")
    print(f"   📄 extracted_vocab_freq.json - TOP100频繁词")

    return output_dir

def print_vocab_summary(standard_vocab):
    """打印词汇摘要"""

    print("\n" + "="*70)
    print("📊 ATC词汇库统计摘要")
    print("="*70)

    for category, words in standard_vocab.items():
        category_name = {
            'weather': '🌦️  天气相关',
            'flight_ops': '✈️  飞行操作',
            'aircraft_parts': '🔧 飞机部件',
            'directions': '🧭 方向和度数',
            'numbers': '🔢 数字相关',
            'communication': '📢 通话用语',
            'airports': '🛫 机场地标',
            'misc': '📋 其他'
        }.get(category, category)

        print(f"\n{category_name} ({len(words)} 词)")
        # 每5个词一行显示
        for i in range(0, len(words), 5):
            print(f"   {', '.join(words[i:i+5])}")

def main():
    print("🎯 构建高质量ATC词汇库（单词级别）")
    print("="*70)

    # 第1步：提取句子
    print("\n📖 第1步：提取训练数据...")
    sentences = extract_sentences_from_dataset()
    print(f"✅ 提取 {len(sentences)} 条句子")

    if not sentences:
        print("❌ 没有提取到任何句子，检查数据目录")
        return

    # 第2步：分词和过滤
    print("\n🔪 第2步：分词和过滤...")
    word_counter, char_set = segment_and_filter_words(sentences)

    # 第3步：构建词汇库
    print("\n🏗️  第3步：构建词汇库...")
    extracted_words = build_quality_vocab(word_counter, min_frequency=2)

    # 第4步：合并标准词汇
    print("\n🔗 第4步：合并标准词汇...")
    merged_vocab, standard_vocab = merge_with_standard_vocab(extracted_words, ATC_STANDARD_VOCAB)

    # 第5步：保存资源
    print("\n💾 第5步：保存资源...")
    output_dir = save_vocab_resources(merged_vocab, ATC_STANDARD_VOCAB, extracted_words)

    # 第6步：打印摘要
    print_vocab_summary(ATC_STANDARD_VOCAB)

    print("\n✨ 完成！")
    print("\n📚 词汇库特点:")
    print("   ✅ 单词级别（不是句子）")
    print("   ✅ 词性过滤（只保留关键词性）")
    print("   ✅ 频率过滤（过滤低频误识别）")
    print("   ✅ 手动标准词汇（确保ATC完整性）")
    print("   ✅ 强泛化能力（可处理新句子）")

    print("\n🚀 下一步:")
    print("   python train_with_vocab_constraint.py")

if __name__ == "__main__":
    main()
