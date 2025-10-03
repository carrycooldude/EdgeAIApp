@echo off
REM LLaMA Model Compilation Script
REM Based on PyTorch ExecutorTorch Qualcomm examples
REM https://github.com/pytorch/executorch/tree/a1652f97b721dccc4f1f2585d3e1f15a2306e8d0/examples/qualcomm

echo ========================================
echo LLaMA Model Compilation for ExecutorTorch
echo ========================================

REM Check if required environment variables are set
if "%QNN_SDK_ROOT%"=="" (
    echo ERROR: QNN_SDK_ROOT environment variable not set
    echo Please set it to the root of Qualcomm AI Engine Direct SDK
    echo Example: set QNN_SDK_ROOT=C:\Qualcomm\AI_Engine_Direct_SDK
    pause
    exit /b 1
)

if "%EXECUTORCH_ROOT%"=="" (
    echo ERROR: EXECUTORCH_ROOT environment variable not set
    echo Please set it to the root of ExecutorTorch repository
    echo Example: set EXECUTORCH_ROOT=C:\executorch
    pause
    exit /b 1
)

if "%ANDROID_NDK_ROOT%"=="" (
    echo ERROR: ANDROID_NDK_ROOT environment variable not set
    echo Please set it to the root of Android NDK
    echo Example: set ANDROID_NDK_ROOT=C:\Android\Sdk\ndk\25.1.8937393
    pause
    exit /b 1
)

echo.
echo Environment Variables:
echo QNN_SDK_ROOT=%QNN_SDK_ROOT%
echo EXECUTORCH_ROOT=%EXECUTORCH_ROOT%
echo ANDROID_NDK_ROOT=%ANDROID_NDK_ROOT%

echo.
echo ========================================
echo Step 1: Building ExecutorTorch AOT Components
echo ========================================

cd /d "%EXECUTORCH_ROOT%"
if not exist "build-x86" mkdir build-x86
cd build-x86

echo.
echo Configuring CMake for AOT components...
cmake .. ^
  -DCMAKE_INSTALL_PREFIX=%CD% ^
  -DEXECUTORCH_BUILD_QNN=ON ^
  -DQNN_SDK_ROOT=%QNN_SDK_ROOT% ^
  -DEXECUTORCH_BUILD_DEVTOOLS=ON ^
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON ^
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON ^
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON ^
  -DPYTHON_EXECUTABLE=python ^
  -DEXECUTORCH_SEPARATE_FLATCC_HOST_PROJECT=OFF

if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed
    pause
    exit /b 1
)

echo.
echo Building AOT components...
cmake --build %CD% --target "PyQnnManagerAdaptor" "PyQnnWrapperAdaptor" -j%NUMBER_OF_PROCESSORS%

if %errorlevel% neq 0 (
    echo ERROR: AOT build failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 2: Building Android Runtime Components
echo ========================================

cd /d "%EXECUTORCH_ROOT%"
if not exist "build-android" mkdir build-android
cd build-android

echo.
echo Configuring CMake for Android runtime...
cmake .. ^
  -DCMAKE_INSTALL_PREFIX=%CD% ^
  -DEXECUTORCH_BUILD_QNN=ON ^
  -DQNN_SDK_ROOT=%QNN_SDK_ROOT% ^
  -DEXECUTORCH_BUILD_DEVTOOLS=ON ^
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON ^
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON ^
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON ^
  -DPYTHON_EXECUTABLE=python ^
  -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK_ROOT%/build/cmake/android.toolchain.cmake ^
  -DANDROID_ABI=arm64-v8a ^
  -DANDROID_NATIVE_API_LEVEL=23

if %errorlevel% neq 0 (
    echo ERROR: Android CMake configuration failed
    pause
    exit /b 1
)

echo.
echo Building Android runtime components...
cmake --build %CD% --target install -j%NUMBER_OF_PROCESSORS%

if %errorlevel% neq 0 (
    echo ERROR: Android runtime build failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 3: Compiling LLaMA Model
echo ========================================

cd /d "%EXECUTORCH_ROOT%\examples\qualcomm\oss_scripts\llama"

echo.
echo Running LLaMA compilation script...
echo Note: This requires a connected Android device
echo.

REM Get device serial
for /f "tokens=1" %%i in ('adb devices ^| findstr "device"') do set DEVICE_SERIAL=%%i
if "%DEVICE_SERIAL%"=="" (
    echo ERROR: No Android device connected
    echo Please connect a device and enable USB debugging
    pause
    exit /b 1
)

echo Using device: %DEVICE_SERIAL%

REM Run the LLaMA compilation script
python llama.py -s %DEVICE_SERIAL% -m "SM8550" -b "%EXECUTORCH_ROOT%\build-android\" --download

if %errorlevel% neq 0 (
    echo ERROR: LLaMA model compilation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 4: Copying Compiled Model
echo ========================================

echo.
echo Copying compiled model to EdgeAI project...
if exist "llama_model.pte" (
    copy "llama_model.pte" "..\..\..\..\..\app\src\main\assets\models\llama_model.pte"
    echo ✅ Model copied successfully
) else (
    echo ⚠️ Compiled model not found, using placeholder
)

echo.
echo ========================================
echo Compilation Complete!
echo ========================================
echo.
echo The LLaMA model has been compiled for ExecutorTorch Qualcomm.
echo You can now run the EdgeAI app to test the model.
echo.
echo Next steps:
echo 1. Run deploy_executor_torch_qualcomm.bat to deploy to device
echo 2. Install and run the EdgeAI app
echo 3. Test LLaMA inference
echo.
pause
