param(
  [string]$InDir = "$PSScriptRoot\watch_in",
  [string]$OutDir = "$PSScriptRoot\outputs\watch_out",
  [int]$Parts = 3,
  [int]$Overlap = 60,
  [int]$MaxNewTokens = 768,
  [switch]$Offload,
  [string]$OffloadFolder = "$PSScriptRoot\offload",
  [ValidateSet('none','4bit')][string]$Quant = 'none'
)
$env:HF_HOME = "$PSScriptRoot\hf_cache"
$env:HUGGINGFACE_HUB_CACHE = "$PSScriptRoot\hf_cache\hub"
$env:PYTHONIOENCODING = 'utf-8'
$py = "$PSScriptRoot\.venv\Scripts\python.exe"
$argv = @(
  (Join-Path $PSScriptRoot 'watch_folder.py'),
  '--in', $InDir,
  '--out', $OutDir,
  '--parts', $Parts,
  '--overlap', $Overlap,
  '--max-new-tokens', $MaxNewTokens,
  '--quant', $Quant
)
if ($Offload) { $argv += @('--offload','--offload-folder', $OffloadFolder, '--max-vram-gib','7') }
& $py @argv
