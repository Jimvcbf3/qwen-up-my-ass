import os
import sys
import argparse
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def parse_args():
    ap = argparse.ArgumentParser("Qwen2-VL read text from image")
    ap.add_argument("--image", required=True, help="Path to PNG/JPG image")
    ap.add_argument("--prompt", default=(
        "Read all visible English and Chinese text from this UI screenshot. "
        "Output plain text lines only, no extra commentary."
    ))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--quant", choices=["none", "4bit"], default="4bit", help="Use 4-bit quant on GPU to avoid OOM")
    ap.add_argument("--offload", action='store_true', help="Enable CPU/GPU offload (no quant)")
    ap.add_argument("--offload-folder", default=None, help="Folder for offloaded weights")
    ap.add_argument("--max-vram-gib", type=int, default=7)
    ap.add_argument("--out", default=None, help="Optional path to save the text output (UTF-8)")
    return ap.parse_args()


@torch.inference_mode()
def run_qwen(image_path: str, prompt: str, max_new_tokens: int = 256, quant: str = "4bit", offload: bool = False, offload_folder: str | None = None, max_vram_gib: int = 7) -> str:
    # Honor D: caches if set by caller
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None

    bnb_config = None
    if use_cuda and (not offload) and quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    processor = AutoProcessor.from_pretrained(model_id)
    if offload:
        max_memory = {0: f"{max_vram_gib}GiB", "cpu": "28GiB"}
        kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16, max_memory=max_memory)
        if offload_folder:
            os.makedirs(offload_folder, exist_ok=True)
            kwargs["offload_folder"] = offload_folder
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    elif bnb_config is not None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb_config, device_map=device_map
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map
        )

    image = load_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # Try to decode only the generated continuation
    input_len = inputs["input_ids"].shape[1]
    cont_ids = gen_ids[0, input_len:]
    out = processor.decode(cont_ids, skip_special_tokens=True)
    return out.strip()


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # ensure UTF-8 console
    except Exception:
        pass
    result = run_qwen(args.image, args.prompt, args.max_new_tokens, args.quant, args.offload, args.offload_folder, args.max_vram_gib)
    # Save if requested
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(result)
    # Print safely (best-effort)
    try:
        print(result)
    except Exception:
        try:
            sys.stdout.buffer.write(result.encode('utf-8', errors='ignore'))
            sys.stdout.buffer.write(b"\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
