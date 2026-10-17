param(
    [string]$Python = "",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [string]$IndexUrl = "https://pypi.org/simple",
    [string]$TorchIndexUrl = "",
    [string]$TorchWheel = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot
$TaskTemp = Join-Path $PSScriptRoot ".work\temp"
[void](New-Item -ItemType Directory -Path $TaskTemp -Force)

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    $SavedTemp = [Environment]::GetEnvironmentVariable("TEMP", "Process")
    $SavedTmp = [Environment]::GetEnvironmentVariable("TMP", "Process")
    try {
        $env:TEMP = $TaskTemp
        $env:TMP = $TaskTemp
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed (exit $LASTEXITCODE): $Executable $($Arguments -join ' ')"
        }
    } finally {
        [Environment]::SetEnvironmentVariable("TEMP", $SavedTemp, "Process")
        [Environment]::SetEnvironmentVariable("TMP", $SavedTmp, "Process")
    }
}

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    if (-not $Python) {
        $CondaPython = Join-Path $env:USERPROFILE "Miniconda3\python.exe"
        if (Test-Path -LiteralPath $CondaPython -PathType Leaf) {
            $Python = $CondaPython
        } else {
            $PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
            if ($null -eq $PythonCommand) {
                throw "Python 3.12 was not found. Pass -Python with its full executable path."
            }
            $Python = $PythonCommand.Source
        }
    }
    Invoke-Checked -Executable $Python -Arguments @(
        "-c", "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'; print(sys.executable)"
    )
    Invoke-Checked -Executable $Python -Arguments @("-m", "venv", (Join-Path $PSScriptRoot ".venv"))
}

Invoke-Checked -Executable $VenvPython -Arguments @(
    "-c", "import sys; assert sys.version_info[:2] == (3, 12), 'Existing .venv must use Python 3.12'; print(sys.executable)"
)
Invoke-Checked -Executable $VenvPython -Arguments @(
    "-m", "pip", "install", "--no-cache-dir", "-r", "requirements-nef.txt", "--index-url", $IndexUrl
)
if (-not $TorchIndexUrl) {
    $TorchIndexUrl = if ($Device -eq "cuda") {
        "https://download.pytorch.org/whl/cu128"
    } else {
        "https://download.pytorch.org/whl/cpu"
    }
}
$TorchVersion = if ($Device -eq "cuda") { "2.8.0+cu128" } else { "2.8.0+cpu" }
if ($TorchWheel) {
    if (-not (Test-Path -LiteralPath $TorchWheel -PathType Leaf)) {
        throw "Local PyTorch wheel was not found: $TorchWheel"
    }
    $WheelFile = Get-Item -LiteralPath $TorchWheel
    if (-not $WheelFile.Name.StartsWith("torch-$TorchVersion-")) {
        throw "Expected a torch-$TorchVersion wheel, got $($WheelFile.Name)"
    }
    Invoke-Checked -Executable $VenvPython -Arguments @(
        "-m", "pip", "install", "--no-cache-dir", $WheelFile.FullName, "--index-url", $IndexUrl
    )
} else {
    Invoke-Checked -Executable $VenvPython -Arguments @(
        "-m", "pip", "install", "--no-cache-dir", "torch==$TorchVersion", "--index-url", $TorchIndexUrl
    )
}

foreach ($Weight in @("low", "high", "colorfix")) {
    $WeightPath = Join-Path $PSScriptRoot "re\weights\$Weight.npz"
    if (-not (Test-Path -LiteralPath $WeightPath -PathType Leaf)) {
        throw "Required StyleTrans weight is missing: $WeightPath"
    }
}

$Check = @'
import sys
import tkinter
import cv2, numpy, rawpy, tifffile, PIL, psutil, torch
from m9.nef import process_nef
device = sys.argv[1]
if torch.__version__ != sys.argv[2]:
    raise RuntimeError(f'Expected PyTorch {sys.argv[2]}, got {torch.__version__}')
if device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA requested, but this Python environment cannot use CUDA')
tensor = torch.arange(1024, dtype=torch.float32, device=device)
value = (tensor * tensor).sum().item()
if device == 'cuda':
    torch.cuda.synchronize()
if not abs(value - 357389824.0) <= 128.0:
    raise RuntimeError(f'Unexpected device tensor result: {value}')
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__, 'device:', device, 'tensor sum:', value)
print('rawpy:', rawpy.__version__, 'LibRaw:', rawpy.libraw_version)
print('Dependency/device check passed. Full StyleTrans validation still requires an actual pipeline run.')
'@
Invoke-Checked -Executable $VenvPython -Arguments @("-c", $Check, $Device, $TorchVersion)
Write-Host "Ready: double-click launch_nef_gui.cmd, or use process_nef.cmd."
Write-Host "Adobe DNG Converter must be installed separately for method=dng."
