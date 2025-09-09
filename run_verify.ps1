param(
  [Parameter(Mandatory=$true)][string]$Image,
  [Parameter(Mandatory=$true)][string]$OcrJson,
  [Parameter(Mandatory=$false)][string]$OutJson = "verified.json",
  [Parameter(Mandatory=$false)][string]$Model = "Qwen/Qwen2-VL-7B-Instruct"
)
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:HF_HOME = Join-Path $proj 'hf_cache'
$env:TRANSFORMERS_CACHE = $env:HF_HOME
$env:HF_DATASETS_CACHE = $env:HF_HOME
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$py = Join-Path $proj '.venv/\Scripts/python.exe'
& $py (Join-Path $proj 'verify_ui_text.py') --image $Image --json $OcrJson --out $OutJson --model $Model --only-flagged
