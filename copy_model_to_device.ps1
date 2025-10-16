# Copy Llama Model to Android Device Script
Write-Host "Setting up Llama-3-8b-chat-hf model on Android device" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green

# Check if ADB is available
try {
    $adbVersion = adb version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "ADB found" -ForegroundColor Green
    } else {
        Write-Host "ADB not found. Please install Android SDK Platform Tools" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "ADB not found. Please install Android SDK Platform Tools" -ForegroundColor Red
    exit 1
}

# Check if device is connected
Write-Host ""
Write-Host "Checking connected devices..." -ForegroundColor Yellow
$devices = adb devices
if ($devices -match "device$") {
    Write-Host "Android device connected" -ForegroundColor Green
} else {
    Write-Host "No Android device connected" -ForegroundColor Red
    Write-Host "Please connect your Android device via USB and enable USB Debugging" -ForegroundColor Cyan
    exit 1
}

# Create directory structure on device
Write-Host ""
Write-Host "Creating directory structure on device..." -ForegroundColor Yellow
$appDir = "/sdcard/Android/data/com.example.edgeai/files"
$modelDir = "$appDir/models/Llama-3-8b-chat-hf"

adb shell "mkdir -p $modelDir"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Directory structure created" -ForegroundColor Green
} else {
    Write-Host "Failed to create directory structure" -ForegroundColor Red
    exit 1
}

# Check if model file exists
$modelFile = "external_models\Llama-3-8b-chat-hf\consolidated.00.pth"
if (Test-Path $modelFile) {
    Write-Host "Model file found: $modelFile" -ForegroundColor Green
    $fileSize = (Get-Item $modelFile).Length
    $fileSizeGB = [math]::Round($fileSize / 1GB, 2)
    Write-Host "File size: $fileSizeGB GB" -ForegroundColor Cyan
} else {
    Write-Host "Model file not found: $modelFile" -ForegroundColor Red
    exit 1
}

# Copy model file to device
Write-Host ""
Write-Host "Copying model file to device..." -ForegroundColor Yellow
Write-Host "This may take several minutes due to the large file size..." -ForegroundColor Cyan

adb push $modelFile $modelDir
if ($LASTEXITCODE -eq 0) {
    Write-Host "Model file copied successfully!" -ForegroundColor Green
} else {
    Write-Host "Failed to copy model file" -ForegroundColor Red
    exit 1
}

# Copy tokenizer files
Write-Host ""
Write-Host "Copying tokenizer files..." -ForegroundColor Yellow

$tokenizerSource = "app\src\main\assets\tokenizer\tokenizer.model"
$tokenizerDest = "$appDir/tokenizer"

adb shell "mkdir -p $tokenizerDest"
adb push $tokenizerSource $tokenizerDest

if ($LASTEXITCODE -eq 0) {
    Write-Host "Tokenizer copied successfully!" -ForegroundColor Green
} else {
    Write-Host "Tokenizer copy failed, but continuing..." -ForegroundColor Yellow
}

# Copy configuration files
Write-Host ""
Write-Host "Copying configuration files..." -ForegroundColor Yellow

$configFiles = @(
    "app\src\main\assets\models\Llama-3-8b-chat-hf\params.json",
    "app\src\main\assets\models\Llama-3-8b-chat-hf\checklist.chk"
)

foreach ($configFile in $configFiles) {
    if (Test-Path $configFile) {
        adb push $configFile $modelDir
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Copied: $(Split-Path $configFile -Leaf)" -ForegroundColor Green
        }
    }
}

# Verify files on device
Write-Host ""
Write-Host "Verifying files on device..." -ForegroundColor Yellow
$deviceFiles = adb shell "ls -la $modelDir"
Write-Host "Files on device:" -ForegroundColor Cyan
Write-Host $deviceFiles -ForegroundColor Gray

# Set permissions
Write-Host ""
Write-Host "Setting file permissions..." -ForegroundColor Yellow
adb shell "chmod -R 755 $appDir"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Permissions set successfully" -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host ""
Write-Host "Your Llama-3-8b-chat-hf model is now ready on your Android device!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Launch the EdgeAI app on your device" -ForegroundColor Cyan
Write-Host "2. The app will automatically detect the model files" -ForegroundColor Cyan
Write-Host "3. Test with sample prompts like 'What is baseball?'" -ForegroundColor Cyan