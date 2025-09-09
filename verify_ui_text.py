import os, json, argparse
from typing import List, Dict, Any
from PIL import Image
import torch
from transformers import AutoProcessor
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

# Ensure caches are on D:
BASE = os.path.dirname(__file__)
CACHE = os.path.join(BASE, 'hf_cache')
os.environ.setdefault('HF_HOME', CACHE)
os.environ.setdefault('TRANSFORMERS_CACHE', CACHE)
os.environ.setdefault('HF_DATASETS_CACHE', CACHE)

PROMPT = (
    "You are verifying OCR for a small UI text crop. "
    "Candidate: '{candidate}'. "
    "Answer strictly as JSON: {\"accept\": true|false, \"corrected\": \"...\"}. "
    "If correct, accept=true and corrected=candidate. If not, accept=false and corrected=what you read."
)

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')

def crop_xyxy(img: Image.Image, xyxy: List[int]) -> Image.Image:
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    x1=max(0,x1); y1=max(0,y1); x2=max(x1+1,x2); y2=max(y1+1,y2)
    return img.crop((x1,y1,x2,y2))

def bbox_to_xyxy(bbox: List[List[int]]) -> List[int]:
    xs=[p[0] for p in bbox]; ys=[p[1] for p in bbox]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

@torch.inference_mode()
def query(model, processor, image: Image.Image, candidate: str) -> Dict[str, Any]:
    messages = [{
        "role":"user",
        "content":[{"type":"image","image":image}, {"type":"text","text": PROMPT.format(candidate=candidate)}]
    }]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt').to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=96)
    out = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    s = out.find('{'); e = out.rfind('}')
    if s!=-1 and e!=-1 and e>s:
        out = out[s:e+1]
    try:
        res = json.loads(out)
        return {"accept": bool(res.get("accept", False)), "corrected": str(res.get("corrected", ""))}
    except Exception:
        return {"accept": False, "corrected": ""}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen2-VL-7B-Instruct')
    ap.add_argument('--device', default='auto', choices=['auto','cuda','cpu'])
    ap.add_argument('--only-flagged', action='store_true')
    ap.add_argument('--conf-threshold', type=float, default=0.6)
    args = ap.parse_args()

    img = load_image(args.image)
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dets = data.get('detections', [])

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = 'auto' if args.device=='auto' else None
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=device_map
    )
    processor = AutoProcessor.from_pretrained(args.model)

    out_dets = []
    for det in dets:
        cand = str(det.get('text',''))
        conf = float(det.get('confidence',0.0) or 0.0)
        xyxy = det.get('xyxy')
        if xyxy is None and det.get('bbox'):
            xyxy = bbox_to_xyxy(det['bbox'])
        if xyxy is None:
            out_dets.append({**det, 'accepted': False, 'verified_text': cand, 'reason': 'no_bbox'})
            continue
        if args.only-flagged and conf >= args.conf_threshold:
            out_dets.append({**det, 'accepted': True, 'verified_text': cand, 'reason': 'high_conf'})
            continue
        crop = crop_xyxy(img, xyxy)
        verdict = query(model, processor, crop, cand)
        accepted = verdict.get('accept', False)
        corrected = verdict.get('corrected', '')
        verified = cand if accepted else (corrected or cand)
        out_dets.append({**det, 'accepted': accepted, 'verified_text': verified})

    out = {
        'image': data.get('image'),
        'model': args.model,
        'only_flagged': args.only_flagged,
        'conf_threshold': args.conf_threshold,
        'count': len(out_dets),
        'detections': out_dets,
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[save] {args.out}")

if __name__ == '__main__':
    main()
