# ExecuTorch Llama-3-8b-chat-hf Setup Test Script (Windows PowerShell)
Write-Host "Testing ExecuTorch Llama-3-8b-chat-hf Setup" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green

# Check if required directories exist
$requiredDirs = @(
    "app\src\main\assets\models\Llama-3-8b-chat-hf",
    "app\src\main\assets\tokenizer",
    "app\src\main\assets\context_binaries"
)

Write-Host "Checking required directories..." -ForegroundColor Cyan
foreach ($dir in $requiredDirs) {
    if (Test-Path $dir) {
        Write-Host "Found: $dir" -ForegroundColor Green
    } else {
        Write-Host "Missing: $dir" -ForegroundColor Red
    }
}

# Check for required files
Write-Host ""
Write-Host "Checking for required files..." -ForegroundColor Cyan

# Check for tokenizer
if (Test-Path "app\src\main\assets\tokenizer\tokenizer.model") {
    Write-Host "Found: tokenizer.model" -ForegroundColor Green
} else {
    Write-Host "Missing: tokenizer.model" -ForegroundColor Red
}

# Check for context binaries
$contextFiles = @(
    "libQnnHtp.so",
    "libQnnHtpV79Stub.so",
    "libQnnSystem.so"
)

foreach ($file in $contextFiles) {
    if (Test-Path "app\src\main\assets\context_binaries\$file") {
        Write-Host "Found: $file" -ForegroundColor Green
    } else {
        Write-Host "Missing: $file" -ForegroundColor Red
    }
}

# Check for Python packages
Write-Host ""
Write-Host "Checking Python packages..." -ForegroundColor Cyan
try {
    $torchVersion = python -c "import torch; print(torch.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Found: PyTorch $torchVersion" -ForegroundColor Green
    } else {
        Write-Host "Missing: PyTorch" -ForegroundColor Red
    }
} catch {
    Write-Host "Missing: PyTorch" -ForegroundColor Red
}

try {
    $transformersVersion = python -c "import transformers; print(transformers.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Found: Transformers $transformersVersion" -ForegroundColor Green
    } else {
        Write-Host "Missing: Transformers" -ForegroundColor Red
    }
} catch {
    Write-Host "Missing: Transformers" -ForegroundColor Red
}

Write-Host ""
Write-Host "Setup verification complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Create Qualcomm AI HUB account at https://aihub.qualcomm.com/" -ForegroundColor Cyan
Write-Host "2. Export Llama-3-8b-chat-hf context binaries from AI HUB" -ForegroundColor Cyan
Write-Host "3. Download tokenizer.model from Meta Llama models" -ForegroundColor Cyan
Write-Host "4. Place files in the appropriate directories" -ForegroundColor Cyan
Write-Host "5. Build and test the app" -ForegroundColor Cyan
