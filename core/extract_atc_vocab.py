"""
提取航空塔台特定词汇并生成误识别纠正表
"""

import json
import csv
from pathlib import Path
from collections import Counter
import difflib

def extract_atc_vocab_from_csv(csv_path):
    """从转录结果CSV中提取常见错误模式"""

    errors = []
    vocab = set()

    # 如果CSV存在，分析错误
    if Path(csv_path).exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ref = row.get('reference_text', '').strip()
                pred = row.get('text', '').strip()

                if ref and pred and ref != pred:
                    errors.append({
                        'reference': ref,
                        'predicted': pred,
                        'similarity': difflib.SequenceMatcher(None, ref, pred).ratio()
                    })

                # 收集参考文本词汇
                if ref:
                    for char in ref:
                        vocab.add(char)

    # 按相似度排序（识别错误但相近的词）
    errors.sort(key=lambda x: x['similarity'])

    return errors, vocab

def create_correction_map(errors_list):
    """创建错误纠正映射"""
    corrections = {}

    for error in errors_list:
        pred = error['predicted'].strip()
        ref = error['reference'].strip()

        # 只保留高度相似但有错误的（0.5-0.95相似度）
        sim = error['similarity']
        if 0.5 < sim < 0.95:
            corrections[pred] = ref

    return corrections

def save_atc_vocab_resources(vocab_set, errors_list, corrections_map, output_dir='./atc_resources'):
    """保存ATC特定资源"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 保存完整词汇表
    with open(output_dir / 'atc_vocab.txt', 'w', encoding='utf-8') as f:
        for char in sorted(vocab_set):
            f.write(char + '\n')

    # 保存错误分析报告
    with open(output_dir / 'error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(errors_list[:50], f, ensure_ascii=False, indent=2)

    # 保存纠正映射
    with open(output_dir / 'correction_map.json', 'w', encoding='utf-8') as f:
        json.dump(corrections_map, f, ensure_ascii=False, indent=2)

    print(f"✅ 词汇资源已保存到 {output_dir}")
    print(f"   - atc_vocab.txt: {len(vocab_set)} 个字符")
    print(f"   - error_analysis.json: 前50个错误样本")
    print(f"   - correction_map.json: {len(corrections_map)} 个纠正规则")

    # 打印top错误
    print("\n📊 TOP 10 错误模式:")
    for i, error in enumerate(errors_list[:10], 1):
        print(f"  {i}. '{error['predicted']}' → '{error['reference']}' (相似度: {error['similarity']:.2%})")

    return output_dir

def main():
    csv_path = "./results/transcription_results.csv"

    print("🔍 分析推理结果中的错误模式...")
    errors, vocab = extract_atc_vocab_from_csv(csv_path)
    print(f"   找到 {len(errors)} 个错误, {len(vocab)} 个字符")

    print("\n🛠️ 创建纠正映射...")
    corrections = create_correction_map(errors)
    print(f"   生成 {len(corrections)} 个纠正规则")

    print("\n💾 保存资源...")
    save_atc_vocab_resources(vocab, errors, corrections)

    print("\n✨ 完成！")
    print("现在可以用词汇约束推理了:")
    print("  python inference.py --vocab_constraint atc_resources/atc_vocab.txt --use_processed")

if __name__ == "__main__":
    main()
