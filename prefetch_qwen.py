from transformers import AutoProcessor
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

m = "Qwen/Qwen2-VL-7B-Instruct"
print("[prefetch] processor", m, flush=True)
AutoProcessor.from_pretrained(m)
print("[prefetch] model", m, flush=True)
Qwen2VLForConditionalGeneration.from_pretrained(m, device_map='cpu')
print("[done] prefetch complete", flush=True)
