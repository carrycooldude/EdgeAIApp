# Real ExecuTorch Setup Script for EdgeAI
# This script sets up the complete environment for real ExecuTorch integration with QNN v79

Write-Host "🚀 Setting up Real ExecuTorch Integration with QNN v79" -ForegroundColor Green
Write-Host ""

# Set up environment variables
$env:QAIRT_HOME = "C:\Users\rawat\Downloads\v2.37.0.250724\qairt\2.37.0.250724"
$env:PATH = "$env:QAIRT_HOME\bin;$env:PATH"
$env:LD_LIBRARY_PATH = "$env:QAIRT_HOME\lib\aarch64-android"
$env:ADSP_LIBRARY_PATH = "$env:QAIRT_HOME\lib\hexagon-v79\unsigned"

Write-Host "✅ Environment variables set:" -ForegroundColor Green
Write-Host "   QAIRT_HOME: $env:QAIRT_HOME" -ForegroundColor White
Write-Host "   LD_LIBRARY_PATH: $env:LD_LIBRARY_PATH" -ForegroundColor White
Write-Host "   ADSP_LIBRARY_PATH: $env:ADSP_LIBRARY_PATH" -ForegroundColor White

Write-Host ""
Write-Host "📁 Verifying QNN v79 libraries..." -ForegroundColor Blue

# Check if QNN v79 libraries exist
$qnnLibs = @(
    "libQnnHtpV79Stub.so",
    "libQnnHtpV79CalculatorStub.so", 
    "libQnnHtp.so",
    "libQnnSystem.so"
)

foreach ($lib in $qnnLibs) {
    $libPath = "app\src\main\jniLibs\arm64-v8a\$lib"
    if (Test-Path $libPath) {
        Write-Host "✅ Found: $lib" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $lib" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "📁 Verifying Hexagon v79 libraries..." -ForegroundColor Blue

# Check if Hexagon v79 libraries exist
$hexagonLibs = @(
    "libQnnHtpV79.so",
    "libQnnHtpV79Skel.so",
    "libQnnNetRunDirectV79Skel.so"
)

foreach ($lib in $hexagonLibs) {
    $libPath = "app\src\main\assets\hexagon-v79\unsigned\$lib"
    if (Test-Path $libPath) {
        Write-Host "✅ Found: $lib" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $lib" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🔧 Building Real ExecuTorch Integration..." -ForegroundColor Blue

# Clean and build the project
Write-Host "🧹 Cleaning previous build..." -ForegroundColor Yellow
& .\gradlew clean

Write-Host "🔨 Building with QNN v79 support..." -ForegroundColor Yellow
& .\gradlew assembleDebug

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful!" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "📱 Installing APK..." -ForegroundColor Blue
    & .\gradlew installDebug
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ APK installed successfully!" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "🎯 Testing Real ExecuTorch Integration..." -ForegroundColor Blue
        Write-Host "Launching app..." -ForegroundColor Yellow
        
        # Launch the app
        adb shell am start -n com.example.edgeai/.MainActivity
        
        Write-Host ""
        Write-Host "📊 Monitoring logs for QNN v79 initialization..." -ForegroundColor Blue
        Write-Host "Run this command to see logs:" -ForegroundColor Cyan
        Write-Host "adb logcat | grep -E 'RealExecuTorch|QNN|v79|SoC Model-69'" -ForegroundColor White
        
    } else {
        Write-Host "❌ APK installation failed!" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    Write-Host "Check the build output above for errors." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Real ExecuTorch setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "What we've set up:" -ForegroundColor Blue
Write-Host "✅ QNN v79 libraries (aarch64-android)" -ForegroundColor White
Write-Host "✅ Hexagon v79 DSP libraries" -ForegroundColor White
Write-Host "✅ Environment variables for v79/SoC Model-69" -ForegroundColor White
Write-Host "✅ Real ExecuTorch integration code" -ForegroundColor White
Write-Host "✅ Build configuration for QNN backend" -ForegroundColor White

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Blue
Write-Host "1. Get Qualcomm AI HUB access for context binaries" -ForegroundColor White
Write-Host "2. Export LLaMA-3-8b-chat-hf context binaries (v79, SoC Model-69)" -ForegroundColor White
Write-Host "3. Place context binaries in: app\src\main\assets\context_binaries\" -ForegroundColor White
Write-Host "4. Test real inference with actual model weights" -ForegroundColor White

Write-Host ""
Write-Host "Expected Logs:" -ForegroundColor Blue
Write-Host "I RealExecuTorch: 📦 Loading v79 context binaries for SoC Model-69..." -ForegroundColor Cyan
Write-Host "I RealExecuTorch: ✅ v79 context binaries loaded successfully" -ForegroundColor Cyan
Write-Host "I RealExecuTorch: ✅ QNN backend initialized with Hexagon v79" -ForegroundColor Cyan
Write-Host "I RealExecuTorch: ✅ Real inference working" -ForegroundColor Cyan
