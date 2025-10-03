@echo off
REM ExecutorTorch Qualcomm Deployment Script
REM Based on PyTorch ExecutorTorch Qualcomm examples
REM https://github.com/pytorch/executorch/tree/a1652f97b721dccc4f1f2585d3e1f15a2306e8d0/examples/qualcomm

echo ========================================
echo ExecutorTorch Qualcomm Deployment
echo ========================================

REM Check if device is connected
adb devices
if %errorlevel% neq 0 (
    echo ERROR: ADB not found or device not connected
    exit /b 1
)

REM Set device directory as per ExecutorTorch patterns
set DEVICE_DIR=/data/local/tmp/executorch_qualcomm_tutorial/

echo.
echo Creating device directory...
adb shell "mkdir -p %DEVICE_DIR%"

echo.
echo Deploying QNN libraries...
echo (Following ExecutorTorch Qualcomm patterns)

REM Note: In a real implementation, these would be actual QNN libraries
REM from the Qualcomm AI Engine Direct SDK
echo - libQnnHtp.so
adb push "app\src\main\assets\qnn_libs\libQnnHtp.so" "%DEVICE_DIR%libQnnHtp.so"

echo - libQnnSystem.so  
adb push "app\src\main\assets\qnn_libs\libQnnSystem.so" "%DEVICE_DIR%libQnnSystem.so"

echo - libQnnHtpV69Stub.so
adb push "app\src\main\assets\qnn_libs\libQnnHtpV69Stub.so" "%DEVICE_DIR%libQnnHtpV69Stub.so"

echo - libQnnHtpV73Stub.so
adb push "app\src\main\assets\qnn_libs\libQnnHtpV73Stub.so" "%DEVICE_DIR%libQnnHtpV73Stub.so"

echo - qnn_executor_runner
adb push "app\src\main\assets\qnn_libs\qnn_executor_runner" "%DEVICE_DIR%qnn_executor_runner"

echo.
echo Setting execution permissions...
adb shell "chmod +x %DEVICE_DIR%qnn_executor_runner"

echo.
echo Deploying LLaMA model...
adb push "app\src\main\assets\models\llama_model.pte" "%DEVICE_DIR%llama_model.pte"

echo.
echo Setting up environment variables...
adb shell "export LD_LIBRARY_PATH=%DEVICE_DIR%:\$LD_LIBRARY_PATH"

echo.
echo Testing ExecutorTorch Qualcomm integration...
adb shell "cd %DEVICE_DIR% && ./qnn_executor_runner --help"

echo.
echo ========================================
echo Deployment Complete!
echo ========================================
echo.
echo The ExecutorTorch Qualcomm integration is now deployed.
echo You can now run the EdgeAI app to test LLaMA inference.
echo.
echo For real implementation, replace placeholder files with:
echo - Actual QNN libraries from Qualcomm AI Engine Direct SDK
echo - Compiled LLaMA model (.pte file) from ExecutorTorch
echo - Real qnn_executor_runner executable
echo.
pause
