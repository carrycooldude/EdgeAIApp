# EdgeAI CLIP Inference Demo

A complete Android application for running OpenAI CLIP model inference on Qualcomm Snapdragon devices using the EdgeAI/QNN runtime.

## 🎯 Overview

This app demonstrates how to:
- Load and run CLIP DLC models on Qualcomm EdgeAI
- Preprocess images for CLIP inference
- Execute inference using QNN (Qualcomm Neural Network) SDK
- Display and save inference results
- Handle camera capture and gallery selection

## 📱 Features

- **🔧 QNN Runtime Integration**: Utilizes Qualcomm's EdgeAI stack for optimized inference
- **📷 Image Input**: Camera capture or gallery selection
- **🖼️ Image Preprocessing**: Automatic resize, normalization, and format conversion
- **⚡ Hardware Acceleration**: Prioritizes NPU → GPU → CPU for best performance
- **📊 Results Analysis**: Detailed output statistics and tensor analysis
- **💾 Data Export**: Save results to external storage with timestamps
- **🔍 Debugging**: Comprehensive logging for troubleshooting

## 📋 Requirements

### Hardware
- Android device with Qualcomm Snapdragon processor
- Minimum Android API level 24 (Android 7.0)
- Recommended: Snapdragon 8 Gen series for optimal NPU performance

### Software
- Android Studio 4.2+
- NDK r21+
- Qualcomm AI Stack SDK
- QNN SDK (Qualcomm Neural Network)
- Gradle 7.0+

### Model
- `openai_clip.dlc` - CLIP model converted to Qualcomm DLC format

## 🏗️ Project Structure

```
EdgeAI/
├── app/
│   ├── cpp/
│   │   ├── qnn_infer.cpp           # Native QNN implementation
│   │   └── CMakeLists.txt          # Native build configuration
│   └── src/main/
│       ├── assets/models/
│       │   └── openai_clip.dlc     # Your CLIP DLC model (REQUIRED)
│       ├── java/com/example/edgeai/
│       │   ├── MainActivity.kt     # Main UI controller
│       │   └── ml/
│       │       └── CLIPInference.kt # ML inference wrapper
│       ├── jniLibs/arm64-v8a/
│       │   └── [QNN libraries]     # QNN runtime libraries
│       └── res/
│           ├── layout/
│           │   └── activity_main.xml # UI layout
│           └── xml/
│               └── file_paths.xml   # File provider config
```

## 🚀 Quick Start

### 1. Prerequisites Setup

**Install Qualcomm AI Stack:**
```bash
# Download from Qualcomm Developer Network
# Extract QNN SDK to your development machine
export QNN_SDK_ROOT="/path/to/qnn/sdk"
```

**Get CLIP DLC Model:**
- Convert your CLIP model to DLC format using QNN tools
- Or obtain pre-converted `openai_clip.dlc`

### 2. Project Setup

**Clone and Configure:**
```bash
git clone <your-repo>
cd EdgeAI
```

**Add Required Files:**

1. **Place DLC Model:**
   ```bash
   # Copy your CLIP model to assets
   cp openai_clip.dlc app/src/main/assets/models/
   ```

2. **Add QNN Libraries:**
   ```bash
   # Copy QNN libraries to jniLibs
   cp $QNN_SDK_ROOT/lib/android/arm64-v8a/*.so app/src/main/jniLibs/arm64-v8a/
   ```

3. **Update CMakeLists.txt:**
   ```cmake
   # In app/cpp/CMakeLists.txt
   set(QNN_SDK_ROOT "/path/to/your/qnn/sdk")
   ```

### 3. Build and Run

**Using Android Studio:**
1. Open project in Android Studio
2. Connect Snapdragon device via USB
3. Build → Make Project
4. Run → Run 'app'

**Using Command Line:**
```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 📖 Usage Guide

### Step 1: Launch App
- Open "EdgeAI CLIP Inference" on your device
- Grant camera and storage permissions when prompted

### Step 2: Load Image
- **Camera**: Tap "📷 Camera" to capture new image
- **Gallery**: Tap "🖼️ Gallery" to select existing image

### Step 3: Run Inference
- Tap "🚀 Run CLIP Inference" 
- Wait for processing (typically 50-200ms on NPU)
- View results in the text area

### Step 4: Analyze Results
- **Statistics**: Min, max, mean values for each output tensor
- **Sample Values**: First 10 elements of each tensor
- **Performance**: Inference timing and device info

### Step 5: Export Results
- Results automatically saved to `/Android/data/com.example.edgeai/files/Documents/`
- File format: `edgeai_clip_results_YYYYMMDD_HHMMSS.txt`

## 🔧 Configuration

### Model Input Specifications
```kotlin
// CLIP model expects:
- Input size: 224×224×3 (RGB)
- Format: CHW (Channel-Height-Width)
- Normalization: ImageNet mean/std
- Data type: Float32
```

### Performance Tuning
```cpp
// Runtime priority (in C++ code):
1. NPU (Hexagon) - Fastest, lowest power
2. GPU (Adreno) - Good performance
3. CPU (ARM) - Fallback option
```

### Logging Configuration
```kotlin
// Adjust log levels in source:
Log.VERBOSE  // Detailed debugging
Log.DEBUG    // Development info
Log.INFO     // General information
Log.WARN     // Warnings
Log.ERROR    // Errors only
```

## 🐛 Troubleshooting

### Common Issues

**❌ "Native library not found"**
```bash
# Solution: Check QNN libraries in jniLibs
ls app/src/main/jniLibs/arm64-v8a/
# Should contain: libQnn*.so files
```

**❌ "Model not found in assets"**
```bash
# Solution: Verify model placement
ls app/src/main/assets/models/
# Should contain: openai_clip.dlc
```

**❌ "QNN initialization failed"**
```bash
# Check device compatibility:
adb shell getprop ro.board.platform
# Should show: Snapdragon chipset (sm8*, sdm*, etc.)
```

**❌ "Inference returns null"**
```bash
# Check logcat for detailed error:
adb logcat -s EdgeAI_QNN_CLIP CLIPInference EdgeAI_CLIP
```

### Performance Issues

**Slow inference (>1000ms):**
- Check if running on CPU instead of NPU/GPU
- Verify QNN libraries are correctly linked
- Ensure device supports hardware acceleration

**Memory issues:**
- Monitor heap usage during inference
- Check for bitmap memory leaks
- Verify model size is appropriate

### Debug Commands

```bash
# Monitor app logs
adb logcat -s EdgeAI_QNN_CLIP:V CLIPInference:V EdgeAI_CLIP:V

# Check device capabilities
adb shell dumpsys | grep -i qualcomm

# Monitor performance
adb shell top -p $(adb shell pidof com.example.edgeai)

# Clear app data
adb shell pm clear com.example.edgeai
```

## 📊 Expected Performance

### Inference Times (typical)
- **NPU (Hexagon)**: 50-100ms
- **GPU (Adreno)**: 100-200ms  
- **CPU (ARM)**: 500-1000ms

### Memory Usage
- **App RAM**: ~100-200MB
- **Model Size**: ~85MB (CLIP ViT-B/32)
- **Temp Storage**: ~50MB during inference

### Accuracy
- **Image Classification**: Matches original CLIP performance
- **Feature Extraction**: Preserves embedding quality
- **Numerical Precision**: FP16 on GPU/NPU, FP32 on CPU

## 🔧 Advanced Configuration

### Custom Model Integration

To use different CLIP variants:

1. **Update model specifications:**
   ```kotlin
   // In CLIPInference.kt
   private const val INPUT_WIDTH = 336  // For CLIP-L
   private const val INPUT_HEIGHT = 336
   ```

2. **Modify preprocessing:**
   ```kotlin
   // Adjust normalization if needed
   private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
   private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)
   ```

### Runtime Optimization

**For maximum NPU utilization:**
```cpp
// In qnn_infer.cpp
// Set specific NPU configuration
// Enable graph optimization
// Use quantized models when available
```

**For development/debugging:**
```cpp
// Force CPU execution for debugging
// Enable verbose QNN logging
// Add performance profiling
```

## 📚 API Reference

### CLIPInference Class

```kotlin
class CLIPInference(context: Context) {
    fun initialize(): Boolean
    fun runInference(bitmap: Bitmap): Map<String, FloatArray>?
    fun release()
    fun isReady(): Boolean
    fun getInputDimensions(): Triple<Int, Int, Int>
}
```

### Native Methods

```cpp
extern "C" {
    bool nativeInitialize(const char* modelPath);
    jobject nativeRunInference(JNIEnv* env, jfloatArray imageData, int width, int height);
    void nativeRelease();
    jintArray nativeGetInputShape();
    jobjectArray nativeGetOutputInfo();
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Guidelines
- Follow Kotlin coding standards
- Add comprehensive logging
- Include unit tests for new features
- Update documentation for API changes
- Test on multiple Snapdragon devices

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qualcomm**: For the EdgeAI SDK and QNN runtime
- **OpenAI**: For the CLIP model architecture
- **Android Community**: For development tools and libraries

## 📞 Support

### Getting Help
- **Issues**: Open GitHub issue for bugs/features
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check Qualcomm Developer Network

### Useful Resources
- [Qualcomm AI Stack Documentation](https://developer.qualcomm.com/software/qualcomm-ai-stack)
- [QNN SDK Reference](https://developer.qualcomm.com/docs/snpe/index.html)
- [CLIP Model Documentation](https://github.com/openai/CLIP)
- [Android NDK Guide](https://developer.android.com/ndk)

---

**Made with ❤️ for Qualcomm EdgeAI Development**
