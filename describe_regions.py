import os
import sys
import argparse
from typing import List, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def split_vertical(img: Image.Image, parts: int = 3, overlap: int = 40) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    w, h = img.size
    regions = []
    step = h // parts
    for i in range(parts):
        y1 = max(0, i * step - (overlap if i > 0 else 0))
        y2 = min(h, (i + 1) * step + (overlap if i < parts - 1 else 0))
        crop = img.crop((0, y1, w, y2))
        regions.append((crop, (0, y1, w, y2)))
    return regions


PROMPT_BASE = (
    "List EVERY visible text line exactly as seen (English + Chinese) in this UI crop. "
    "Top-to-bottom, left-to-right. Do NOT summarize. One line per item. No emojis or extra symbols."
)


@torch.inference_mode()
def load_model(
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    quant: str = "4bit",
    offload: bool = False,
    offload_folder: str | None = None,
    max_vram_gib: int = 7,
):
    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None
    bnb_config = None
    if use_cuda and not offload and quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    processor = AutoProcessor.from_pretrained(model_id)
    if offload:
        # Full-precision with CPU/GPU offload (no quantization)
        # Accelerate expects integer device ids for GPU keys
        max_memory = {0: f"{max_vram_gib}GiB", "cpu": "28GiB"}
        kwargs = dict(
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory=max_memory,
        )
        if offload_folder:
            os.makedirs(offload_folder, exist_ok=True)
            kwargs["offload_folder"] = offload_folder
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, **kwargs
        )
    elif bnb_config is not None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb_config, device_map=device_map
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map
        )
    return model, processor


@torch.inference_mode()
def run_on_crop(model, processor, crop: Image.Image, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[crop], return_tensors="pt").to(model.device)
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    input_len = inputs["input_ids"].shape[1]
    cont_ids = gen_ids[0, input_len:]
    out = processor.decode(cont_ids, skip_special_tokens=True)
    return out.strip()


def parse_args():
    ap = argparse.ArgumentParser("Qwen2-VL multi-region text reader")
    ap.add_argument("--image", required=True)
    ap.add_argument("--parts", type=int, default=3)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    ap.add_argument("--offload", action="store_true", help="Enable CPU/GPU offload (no quant)")
    ap.add_argument("--offload-folder", default=None, help="Folder for offloaded weights (on D:)")
    ap.add_argument("--max-vram-gib", type=int, default=7, help="VRAM budget for offload")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    img = load_image(args.image)
    regions = split_vertical(img, parts=args.parts, overlap=args.overlap)
    model, processor = load_model(
        quant=args.quant,
        offload=args.offload,
        offload_folder=args.offload_folder,
        max_vram_gib=args.max_vram_gib,
    )

    all_lines: List[str] = []
    seen = set()
    for idx, (crop, box) in enumerate(regions, start=1):
        prompt = PROMPT_BASE
        text = run_on_crop(model, processor, crop, prompt, args.max_new_tokens)
        # normalize into lines
        lines = [ln.strip() for ln in text.splitlines()]
        for ln in lines:
            if not ln:
                continue
            # basic filtering of bullets or numbering the model might add
            ln_clean = ln.lstrip("-•0123456789. ")
            if ln_clean and ln_clean not in seen:
                seen.add(ln_clean)
                all_lines.append(ln_clean)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_lines))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
