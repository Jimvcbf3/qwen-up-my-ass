from transformers import AutoProcessor
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
m = "Qwen/Qwen2-VL-7B-Instruct"
print("[test] loading processor...")
p = AutoProcessor.from_pretrained(m)
print("[ok] processor")
print("[test] loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(m, device_map='cpu')
print("[ok] model")
