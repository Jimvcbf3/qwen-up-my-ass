from transformers import AutoProcessor
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
m = "Qwen/Qwen2-VL-7B-Instruct"
print("[test] loading processor local-only...")
p = AutoProcessor.from_pretrained(m, local_files_only=True)
print("[ok] processor")
print("[test] loading model local-only...")
model = Qwen2VLForConditionalGeneration.from_pretrained(m, device_map='cpu', local_files_only=True)
print("[ok] model")
