"""
测试推理速度 - 对比单条推理和预处理数据推理
"""

import torch
import time
from inference import WhisperInference
from datasets import load_from_disk

# 加载模型
print("加载模型...")
infer = WhisperInference("models/final_model")

# 测试1: 单条原始音频推理
print("\n" + "="*60)
print("测试1: 单条原始音频推理")
print("="*60)
audio_path = r"D:\NPU_works\语音\demo\ATCOSIM\WAVdata\zf3\zf3_02\zf3_02_001.wav"

# 预热
print("预热中...")
_ = infer.transcribe_file(audio_path, language="en")

# 正式测试（连续3次）
times = []
for i in range(3):
    result = infer.transcribe_file(audio_path, language="en")
    times.append(result['inference_time'])
    print(f"第{i+1}次: {result['inference_time']:.3f}s - {result['text']}")

print(f"平均推理时间: {sum(times)/len(times):.3f}s")

# 测试2: 预处理数据推理
print("\n" + "="*60)
print("测试2: 预处理数据推理")
print("="*60)

dataset = load_from_disk("processed_data/test")
sample = dataset[0]

# 预热
print("预热中...")
input_features = torch.tensor(sample["input_features"], dtype=torch.float32).unsqueeze(0).to(infer.device)
with torch.inference_mode():
    with torch.cuda.amp.autocast():
        _ = infer.model.generate(input_features, max_new_tokens=225, num_beams=1, do_sample=False, early_stopping=True)

# 正式测试（连续3次）
times2 = []
for i in range(3):
    t_start = time.time()
    input_features = torch.tensor(sample["input_features"], dtype=torch.float32).unsqueeze(0).to(infer.device)
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            pred_ids = infer.model.generate(input_features, max_new_tokens=225, num_beams=1, do_sample=False, early_stopping=True)
    t_end = time.time()
    text = infer.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
    times2.append(t_end - t_start)
    print(f"第{i+1}次: {t_end - t_start:.3f}s - {text}")

print(f"平均推理时间: {sum(times2)/len(times2):.3f}s")

print("\n" + "="*60)
print("总结")
print("="*60)
print(f"单条原始音频推理: {sum(times)/len(times):.3f}s")
print(f"预处理数据推理: {sum(times2)/len(times2):.3f}s")
print(f"速度差异: {(sum(times)/len(times)) / (sum(times2)/len(times2)):.2f}x")
