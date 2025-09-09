# Agent Playbook — OCR + Qwen Verifier (D: only)

This file tells a future agent (Codex) how to wire the pipeline on this machine. Do NOT write to `C:`. All code, caches, and outputs MUST live on `D:`.

## Ground Rules
- Use only `D:` storage. Respect existing dirs:
  - Project root: `D:\QWEN2-VL7B project`
  - HF cache: `D:\QWEN2-VL7B project\hf_cache` (set `HF_HOME`, `HUGGINGFACE_HUB_CACHE`)
  - Offload dir: `D:\QWEN2-VL7B project\offload`
  - Outputs: `D:\QWEN2-VL7B project\outputs`
- Python venv already exists: `D:\QWEN2-VL7B project\.venv` (Windows).
- GPU: RTX 4060 (8 GB). 7B VLM must run with either 4‑bit OR CPU/GPU offload. Prefer offload for accuracy, 4‑bit for speed.

## What’s Here Already
- Qwen single/multi region readers:
  - `describe_image.py` — single‑image, prompt‑driven read; supports 4‑bit or offload.
  - `describe_regions.py` — splits image (default 3 vertical crops) and merges text; supports 4‑bit or offload.
- Qwen model + cache are present on `D:`.
- Torch CUDA (cu124) installed. BitsAndBytes installed.

## The Desired Pipeline (MCP‑style)
Goal: A long‑running Qwen verifier service, with OCR workers (PaddleOCR primary, EasyOCR secondary) and an orchestrator that fuses results.

### Stages
1) Ingest
   - Input: UI screenshots (PC/phone). Accept folder watch or JSON job files.
   - Location: `D:\queue\in` (create if missing). Jobs reference images via absolute `D:` paths.

2) OCR Workers
   - PaddleOCR (primary) — higher CN accuracy.
     - Expect a callable/module or CLI that returns JSON with: `text`, `confidence`, `xyxy` or `bbox`.
   - EasyOCR (secondary) — run in parallel for cross‑check.
   - Save raw outputs to `D:\queue\work\<job_id>\paddle.json`, `easyocr.json`.

3) Aggregator/Rules
   - Merge by overlapping boxes; for each line compute:
     - `best_text`, `best_conf`, `sources` (paddle/easy), `distance` (Levenshtein), `flagged` if low conf (<0.6), conflict, or regex fail.
   - Write `D:\queue\work\<job_id>\merged.json`.

4) Qwen Verifier (VLM)
   - Long‑running service that stays warm.
   - For each flagged line, crop from the source image and ask Qwen: “Does this read ‘<candidate>’? If no, return corrected.”
   - Update merged JSON with `accepted`, `verified_text`.

5) Output
   - Final: `D:\queue\out\<job_id>\final.json` and optional annotated image.

### Contract (final.json)
```
{
  "image": "<filename>",
  "detections": [
    {
      "xyxy": [x1,y1,x2,y2],
      "text": "paddle_text",
      "confidence": 0.0-1.0,
      "easy_text": "easy_text",
      "easy_conf": 0.0-1.0,
      "best_text": "...",
      "best_conf": 0.0-1.0,
      "flagged": true|false,
      "verified_text": "...",
      "accepted": true|false
    }
  ]
}
```

## Qwen Service (keep it warm)
Implement a small service that loads Qwen once and serves requests.

### Option A: Simple HTTP (FastAPI)
- Endpoint `POST /verify_crop` with body:
```
{
  "image_path": "D:\\...\\screenshot.png",
  "xyxy": [x1,y1,x2,y2],
  "candidate": "text",
  "max_new_tokens": 512
}
```
- Response: `{ "accept": true|false, "corrected": "..." }`
- Start script in project venv, export env:
  - Accuracy (no quant): use offload
    - device_map="auto", max_memory={0:"7GiB", "cpu":"28GiB"}, offload_folder=`D:\QWEN2-VL7B project\offload`
  - Speed (with minor drop): 4‑bit BitsAndBytes (nf4 + double‑quant)

### Option B: Local queue worker
- Watch `D:\queue\verify\in` for crop jobs; write replies to `D:\queue\verify\out`.

## Orchestrator TODOs (for future agent)
1) Create dirs: `D:\queue\in`, `D:\queue\work`, `D:\queue\out`.
2) Implement a job schema: `job.json` with keys `{ id, image, options }`.
3) PaddleOCR/EasyOCR runners (Python or CLI) that output JSON per above.
   - Respect D: only. Cache models on D:.
4) Aggregator: merge + rule flags + crop generator for Qwen.
5) Qwen verifier client: call the warm service for flagged lines.
6) Finalizer: write `final.json` + annotated image.
7) Batch/daemon: a single `watch_queue.py` that never exits (warm performance).

## Environment & Commands
- Always set before running:
  - `HF_HOME=D:\QWEN2-VL7B project\hf_cache`
  - `HUGGINGFACE_HUB_CACHE=D:\QWEN2-VL7B project\hf_cache\hub`
  - `PYTHONIOENCODING=utf-8`

### Start Qwen service (suggested)
```
PS> cd "D:\QWEN2-VL7B project"
PS> .\.venv\Scripts\python.exe qwen_service.py --offload --max-vram-gib 7 --offload-folder "D:\QWEN2-VL7B project\offload"
```

### One-off verify (existing tools)
- Single image read: `describe_image.py` (prompt‑based)
- Multi‑region read: `describe_regions.py` (better recall)

Examples:
```
PS> .\.venv\Scripts\python.exe describe_regions.py `
      --image "D:\paddle_ocr_project\input\test.png" `
      --parts 3 --overlap 60 `
      --max-new-tokens 768 `
      --quant none --offload --max-vram-gib 7 `
      --offload-folder "D:\QWEN2-VL7B project\offload" `
      --out "D:\QWEN2-VL7B project\outputs\qwen_full.txt"
```

## Notes on Accuracy & Performance
- 7B Offload = best accuracy (matches CPU FP), slower than pure GPU but way faster than CPU‑only.
- 7B 4‑bit = fastest on 4060, slight accuracy drop (usually small on UI screenshots).
- For dense UIs, use multi‑region + max_new_tokens 512–768, greedy decoding.

## Do Not
- Do not write any cache/files to `C:`.
- Do not terminate the Qwen service between images if you want warm performance.

## Quick Task List (for new Codex)
1. Add `qwen_service.py` (FastAPI) with offload loader + `/verify_crop`.
2. Implement `watch_queue.py` to orchestrate OCR → aggregate → Qwen verify → final outputs.
3. Add PaddleOCR/EasyOCR runners that save JSON to `D:\queue\work\<job_id>`.
4. Keep everything on `D:` and reuse the project venv.

