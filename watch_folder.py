import os
import sys
import time
import argparse
from typing import List
from PIL import Image

# Pin HF caches to project hf_cache by default (D: only)
BASE = os.path.dirname(__file__)
CACHE = os.path.join(BASE, 'hf_cache')
os.environ.setdefault('HF_HOME', CACHE)
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(CACHE, 'hub'))
os.environ.setdefault('TRANSFORMERS_CACHE', CACHE)
os.environ.setdefault('HF_DATASETS_CACHE', CACHE)
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from describe_regions import (
    load_model,
    split_vertical,
    run_on_crop,
    PROMPT_BASE,
)


def is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def wait_stable(path: str, checks: int = 3, interval: float = 0.5) -> bool:
    """Wait until file size stops changing (basic write-complete heuristic)."""
    last = -1
    for _ in range(checks):
        try:
            cur = os.path.getsize(path)
        except OSError:
            return False
        if cur == last:
            return True
        last = cur
        time.sleep(interval)
    return True


def read_image(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def process_image(model, processor, image_path: str, parts: int, overlap: int, max_new_tokens: int) -> List[str]:
    img = read_image(image_path)
    regions = split_vertical(img, parts=parts, overlap=overlap)
    lines_out: List[str] = []
    seen = set()
    for crop, _box in regions:
        txt = run_on_crop(model, processor, crop, PROMPT_BASE, max_new_tokens)
        for ln in (l.strip() for l in txt.splitlines()):
            if not ln:
                continue
            # strip bullets/numbers the model might add
            clean = ln.lstrip('-•0123456789. ').strip()
            if clean and clean not in seen:
                seen.add(clean)
                lines_out.append(clean)
    return lines_out


def parse_args():
    ap = argparse.ArgumentParser('Watch a folder and read UI text with Qwen2-VL-7B (warm process)')
    ap.add_argument('--in', dest='indir', default=os.path.join(BASE, 'watch_in'), help='Input folder to watch')
    ap.add_argument('--out', dest='outdir', default=os.path.join(BASE, 'outputs', 'watch_out'), help='Output folder for .txt')
    ap.add_argument('--parts', type=int, default=3)
    ap.add_argument('--overlap', type=int, default=60)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    ap.add_argument('--quant', choices=['none', '4bit'], default='none')
    ap.add_argument('--offload', action='store_true', help='Enable CPU/GPU offload (no quant)')
    ap.add_argument('--offload-folder', default=os.path.join(BASE, 'offload'))
    ap.add_argument('--max-vram-gib', type=int, default=7)
    ap.add_argument('--poll', type=float, default=2.0, help='Polling interval seconds')
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    os.makedirs(args.indir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    if args.offload:
        os.makedirs(args.offload_folder or os.path.join(BASE, 'offload'), exist_ok=True)

    print(f"[init] loading Qwen2-VL-7B quant={args.quant} offload={args.offload}")
    model, processor = load_model(
        quant=args.quant,
        offload=args.offload,
        offload_folder=args.offload_folder,
        max_vram_gib=args.max_vram_gib,
    )
    print("[ready] watching:", args.indir)

    seen_files = set()
    while True:
        try:
            for name in os.listdir(args.indir):
                src = os.path.join(args.indir, name)
                if not is_image(src):
                    continue
                out_txt = os.path.join(args.outdir, os.path.splitext(name)[0] + '.txt')
                if src in seen_files and os.path.exists(out_txt):
                    continue
                # Wait until file is stable (avoid half-written file)
                if not wait_stable(src):
                    continue
                print(f"[run] {name}")
                lines = process_image(model, processor, src, args.parts, args.overlap, args.max_new_tokens)
                with open(out_txt, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))
                print(f"[save] {out_txt} ({len(lines)} lines)")
                seen_files.add(src)
        except KeyboardInterrupt:
            print("[exit] stopped by user")
            break
        except Exception as e:
            print(f"[warn] {e}")
        time.sleep(args.poll)


if __name__ == '__main__':
    main()

