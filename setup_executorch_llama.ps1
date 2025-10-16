# ExecuTorch Llama-3-8b-chat-hf Setup Script (Windows PowerShell)
Write-Host "Setting up ExecuTorch Llama-3-8b-chat-hf with Qualcomm HTP backend" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "app/build.gradle.kts")) {
    Write-Host "Error: Please run this script from the EdgeAI project root directory" -ForegroundColor Red
    exit 1
}

Write-Host "Found EdgeAI project directory" -ForegroundColor Green

# Step 1: Install ExecuTorch dependencies
Write-Host ""
Write-Host "Step 1: Installing ExecuTorch dependencies..." -ForegroundColor Yellow

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Python found: $pythonVersion" -ForegroundColor Green
        Write-Host "Installing ExecuTorch packages..." -ForegroundColor Cyan
        
        pip install executorch
        pip install torch
        pip install transformers
        pip install optimum-executorch
        
        Write-Host "ExecuTorch packages installed successfully" -ForegroundColor Green
    } else {
        Write-Host "Python not found. Please install Python and ExecuTorch manually:" -ForegroundColor Yellow
        Write-Host "pip install executorch torch transformers optimum-executorch" -ForegroundColor Cyan
    }
} catch {
    Write-Host "Python not found. Please install Python3 and ExecuTorch manually:" -ForegroundColor Yellow
    Write-Host "pip install executorch torch transformers optimum-executorch" -ForegroundColor Cyan
}

# Step 2: Create directory structure for models
Write-Host ""
Write-Host "Step 2: Creating directory structure for Llama-3-8b-chat-hf model..." -ForegroundColor Yellow

$modelDir = "app\src\main\assets\models\Llama-3-8b-chat-hf"
$tokenizerDir = "app\src\main\assets\tokenizer"
$contextBinariesDir = "app\src\main\assets\context_binaries"

# Create directories
New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
New-Item -ItemType Directory -Path $tokenizerDir -Force | Out-Null
New-Item -ItemType Directory -Path $contextBinariesDir -Force | Out-Null

Write-Host "Created directories:" -ForegroundColor Green
Write-Host "  - $modelDir" -ForegroundColor Cyan
Write-Host "  - $tokenizerDir" -ForegroundColor Cyan
Write-Host "  - $contextBinariesDir" -ForegroundColor Cyan

# Step 3: Setup instructions
Write-Host ""
Write-Host "Step 3: Setup Instructions for Llama-3-8b-chat-hf" -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "To complete the setup, you need to:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Create Qualcomm AI HUB Account:" -ForegroundColor Green
Write-Host "   - Visit: https://aihub.qualcomm.com/" -ForegroundColor Cyan
Write-Host "   - Create an account and get access to AI HUB" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Export Llama-3-8b-chat-hf Context Binaries:" -ForegroundColor Green
Write-Host "   - Follow instructions at: https://huggingface.co/qualcomm/Llama-v3-8B-Chat" -ForegroundColor Cyan
Write-Host "   - Export context binaries using Qualcomm AI HUB" -ForegroundColor Cyan
Write-Host "   - This will take some time to complete" -ForegroundColor Cyan
Write-Host "   - Place the exported binaries in: $contextBinariesDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Download Llama 3 Tokenizer:" -ForegroundColor Green
Write-Host "   - Visit: https://github.com/meta-llama/llama-models/blob/main/README.md" -ForegroundColor Cyan
Write-Host "   - Download tokenizer.model file" -ForegroundColor Cyan
Write-Host "   - Place it in: $tokenizerDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Verify Context Binary Version:" -ForegroundColor Green
Write-Host "   - Ensure context binaries are version v79" -ForegroundColor Cyan
Write-Host "   - Compatible with SoC Model-69" -ForegroundColor Cyan
Write-Host "   - Required files: libQnnHtp.so, libQnnHtpV79Stub.so, libQnnSystem.so" -ForegroundColor Cyan
Write-Host ""

# Step 4: Build instructions
Write-Host ""
Write-Host "Step 4: Build and Test Instructions" -ForegroundColor Yellow
Write-Host "===================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "To build and test the app:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Build the project:" -ForegroundColor Green
Write-Host "   .\gradlew.bat assembleDebug" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Install on device:" -ForegroundColor Green
Write-Host "   .\gradlew.bat installDebug" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Run the app and test with prompt: What is baseball?" -ForegroundColor Green
Write-Host ""

Write-Host "Setup script completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Follow the setup instructions above" -ForegroundColor Cyan
Write-Host "2. Build and test the app" -ForegroundColor Cyan
Write-Host ""
Write-Host "For more information, visit:" -ForegroundColor Cyan
Write-Host "- ExecuTorch: https://github.com/pytorch/executorch" -ForegroundColor Cyan
Write-Host "- Qualcomm AI HUB: https://aihub.qualcomm.com/" -ForegroundColor Cyan
Write-Host "- Llama-3-8b-chat-hf: https://huggingface.co/qualcomm/Llama-v3-8B-Chat" -ForegroundColor Cyan