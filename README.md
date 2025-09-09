# QWEN2-VL7B project

Local verifier for UI OCR using Qwen/Qwen2-VL-7B-Instruct. All caches live on D:.

## Setup
- Python venv: `.venv` (reuses system site-packages to avoid reinstalling torch)
- Install deps: `./.venv/Scripts/pip install -r requirements.txt`
- Keep caches on D:: set `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE` to `./hf_cache`

## Run (example)
```
# PowerShell in project folder
$env:HF_HOME = "$PWD\hf_cache"
$env:TRANSFORMERS_CACHE = "$PWD\hf_cache"
$env:HF_DATASETS_CACHE = "$PWD\hf_cache"
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

./.venv/Scripts/python.exe verify_ui_text.py `
  --image "D:\\paddle_ocr_project\\input\\test.png" `
  --json  "D:\\easyocr_outputs\\run1_ocr.json" `
  --out   "$PWD\\verified.json" `
  --model "Qwen/Qwen2-VL-7B-Instruct" `
  --only-flagged --conf-threshold 0.6
```

Input JSON (example):
```
{
  "image": "test.png",
  "detections": [
    {"text": "Edit", "confidence": 0.42, "xyxy": [100,120,200,160]},
    {"text": "電腦季件二手交易", "confidence": 0.55, "xyxy": [300,420,620,470]}
  ]
}
```
