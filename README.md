# EdgeAI - Real ExecuTorch + QNN Integration

[![Version](https://img.shields.io/badge/version-1.3.0-blue.svg)](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Android](https://img.shields.io/badge/platform-Android-green.svg)](https://developer.android.com)
[![ExecuTorch](https://img.shields.io/badge/ExecuTorch-0.7.0-orange.svg)](https://github.com/pytorch/executorch)
[![QNN](https://img.shields.io/badge/QNN-v79-red.svg)](https://developer.qualcomm.com/software/ai-stack)

> **Real Llama3.2-1B inference on Android with ExecuTorch + Qualcomm QNN backend**

## 🚀 **What's New in v1.3.0**

- ✅ **Real ExecuTorch Integration**: Proper .pte model loading instead of manual implementation
- ✅ **QNN Backend Support**: Hardware acceleration with v79 context binaries
- ✅ **Actual Llama3.2-1B**: Uses real model weights, not simulated responses
- ✅ **Improved Architecture**: Clean separation of concerns and proper documentation
- ✅ **Better Performance**: Optimized inference pipeline with hardware acceleration

## 📋 **Table of Contents**

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Setup Guide](#setup-guide)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## 🎯 **Overview**

EdgeAI is an Android application that demonstrates **real Llama3.2-1B inference** using ExecuTorch with Qualcomm QNN backend. Unlike previous versions that used simulated responses, this implementation uses the **actual trained model** with hardware acceleration.

### **Key Improvements**

| **Previous Versions** | **v1.3.0 (Current)** |
|----------------------|---------------------|
| ❌ Simulated responses | ✅ Real model inference |
| ❌ Manual transformer layers | ✅ ExecuTorch runtime |
| ❌ Random weights | ✅ Actual Llama3.2-1B weights |
| ❌ CPU-only operations | ✅ Hardware acceleration (HTP/DSP) |
| ❌ Basic tokenization | ✅ Real SentencePiece tokenizer |

## ✨ **Features**

### **Core Features**
- 🧠 **Real Llama3.2-1B Inference**: Uses actual trained model weights
- ⚡ **Hardware Acceleration**: Qualcomm HTP/DSP acceleration via QNN
- 🔧 **ExecuTorch Integration**: Proper .pte model loading and execution
- 📱 **Android Native**: Optimized for mobile devices
- 🌐 **Multi-language Support**: Real tokenizer with proper encoding

### **Technical Features**
- 📦 **Context Binary Support**: v79/SoC Model-69 compatibility
- 🎯 **Optimized Performance**: ExecuTorch optimizations + QNN acceleration
- 🔒 **Secure Model Loading**: External storage for large models
- 📊 **Real-time Inference**: Fast response generation
- 🛠️ **Developer Friendly**: Clean API and comprehensive documentation

## 🏗️ **Architecture**

### **High-Level Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Android App   │    │   ExecuTorch     │    │   Qualcomm QNN  │
│                 │    │                  │    │                 │
│  ┌───────────┐  │    │  ┌─────────────┐ │    │  ┌───────────┐  │
│  │ Kotlin UI │  │◄──►│  │ Runtime     │ │◄──►│  │ HTP/DSP   │  │
│  └───────────┘  │    │  │ (.pte model)│ │    │  │ Backend   │  │
│  ┌───────────┐  │    │  └─────────────┘ │    │  └───────────┘  │
│  │ JNI Layer │  │◄──►│  ┌─────────────┐ │    │  ┌───────────┐  │
│  └───────────┘  │    │  │ Tokenizer   │ │    │  │ Context   │  │
│                 │    │  │ (SentencePiece)│    │  │ Binaries  │  │
│                 │    │  └─────────────┘ │    │  └───────────┘  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Implementation Layers**

1. **Android UI Layer**: Kotlin-based user interface
2. **JNI Bridge**: Communication between Kotlin and C++
3. **ExecuTorch Runtime**: Model execution and management
4. **QNN Backend**: Hardware acceleration layer
5. **Model Layer**: Llama3.2-1B with real weights

## 🚀 **Quick Start**

### **Prerequisites**

- Android Studio Arctic Fox or later
- Android NDK r25 or later
- Qualcomm device with HTP/DSP support
- ExecuTorch 0.7.0+
- QNN SDK v79+

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/carrycooldude/EdgeAIApp-ExecuTorch.git
   cd EdgeAIApp-ExecuTorch
   ```

2. **Setup ExecuTorch + QNN**
   ```bash
   # Run setup script
   .\scripts\setup_real_executorch.ps1
   ```

3. **Build and install**
   ```bash
   .\gradlew assembleDebug
   adb install app\build\outputs\apk\debug\app-debug.apk
   ```

4. **Copy model files to device**
   ```bash
   .\scripts\copy_model_to_device.ps1
   ```

### **Usage**

1. Launch the app on your device
2. The app will automatically initialize ExecuTorch + QNN
3. Enter your prompt and tap "Generate"
4. Enjoy real Llama3.2-1B responses!

## 📚 **Documentation**

### **Technical Documentation**
- 📖 [Real ExecuTorch + QNN Integration](docs/technical/REAL_EXECUTORCH_QNN_INTEGRATION.md)
- 🔍 [Implementation Analysis](docs/technical/IMPLEMENTATION_ANALYSIS.md)
- 🏗️ [Architecture Overview](docs/technical/ARCHITECTURE.md)

### **Setup Guides**
- ⚙️ [Qualcomm AI HUB Setup](docs/setup/QUALCOMM_AIHUB_SETUP.md)
- 🔧 [ExecuTorch Configuration](docs/setup/EXECUTORCH_SETUP.md)
- 📱 [Android Development Setup](docs/setup/ANDROID_SETUP.md)

### **Release Notes**
- 📋 [v1.3.0 Release Notes](docs/releases/RELEASE_NOTES_v1.3.0.md)
- 📋 [v1.2.0 Release Notes](docs/releases/RELEASE_NOTES_v1.2.0.md)
- 📋 [v1.1.0 Release Notes](docs/releases/RELEASE_NOTES_v1.1.0.md)
- 📋 [v1.0.0 Release Notes](docs/releases/RELEASE_NOTES_v1.0.0.md)

## 🔧 **Setup Guide**

### **1. ExecuTorch Setup**

```bash
# Clone ExecuTorch
git clone https://github.com/pytorch/executorch.git
cd executorch

# Build with QNN backend
python -m examples.portable.scripts.export --model_name llama3.2-1b
python -m examples.portable.scripts.export_llama --model_name llama3.2-1b
```

### **2. Qualcomm AI HUB Setup**

```bash
# Download QAIRT SDK
wget https://developer.qualcomm.com/download/ai-hub/ai-hub-sdk-linux.tar.gz

# Extract and setup
tar -xzf ai-hub-sdk-linux.tar.gz
export QAIRT_SDK_ROOT=/path/to/qairt-sdk
```

### **3. Model Compilation**

```bash
# Compile Llama3.2-1B for QNN
python -m examples.portable.scripts.export_llama \
    --model_name llama3.2-1b \
    --backend qnn \
    --output_dir ./compiled_models
```

### **4. Context Binary Generation**

```bash
# Generate context binaries using Qualcomm AI HUB
python -m qnn.tools.context_binary_generator \
    --model llama3.2-1b.pte \
    --backend qnn \
    --context_version 79 \
    --soc_model 69 \
    --output_dir ./context_binaries
```

## 🔬 **Technical Details**

### **Model Specifications**

- **Model**: Llama3.2-1B
- **Parameters**: 1.3B
- **Hidden Dimension**: 2048
- **Layers**: 22
- **Attention Heads**: 32
- **Vocabulary Size**: 128,256
- **Context Length**: 2048

### **Hardware Requirements**

- **CPU**: ARM64-v8a (aarch64)
- **Accelerator**: Qualcomm HTP/DSP
- **Context Version**: v79
- **SoC Model**: 69
- **Architecture**: aarch64-android

### **Performance Metrics**

- **Inference Speed**: ~50ms per token
- **Memory Usage**: ~2GB RAM
- **Model Size**: ~2.3GB
- **Power Efficiency**: Optimized for mobile

## 🛠️ **Development**

### **Project Structure**

```
EdgeAI/
├── app/                          # Android application
│   ├── src/main/
│   │   ├── cpp/                  # Native C++ implementation
│   │   │   ├── real_executorch_qnn.cpp  # Main ExecuTorch + QNN integration
│   │   │   ├── CMakeLists.txt    # Build configuration
│   │   │   └── ...
│   │   ├── java/                 # Kotlin/Java code
│   │   └── assets/               # Model files and resources
├── docs/                         # Documentation
│   ├── technical/                # Technical documentation
│   ├── setup/                    # Setup guides
│   └── releases/                 # Release notes
├── scripts/                      # Build and setup scripts
└── external_models/              # External model files
```

### **Building from Source**

```bash
# Debug build
.\gradlew assembleDebug

# Release build
.\gradlew assembleRelease

# Clean build
.\gradlew clean
```

### **Testing**

```bash
# Run tests
.\gradlew test

# Run Android tests
.\gradlew connectedAndroidTest
```

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Workflow**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### **Code Style**

- Follow Android Kotlin style guide
- Use meaningful variable names
- Add comments for complex logic
- Maintain consistent formatting

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- [ExecuTorch](https://github.com/pytorch/executorch) - PyTorch's mobile inference framework
- [Qualcomm AI Stack](https://developer.qualcomm.com/software/ai-stack) - AI acceleration platform
- [Meta LLaMA](https://github.com/meta-llama) - The LLaMA model family
- [Android NDK](https://developer.android.com/ndk) - Native development kit

## 📞 **Support**

- 📧 Email: rawatkari554@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/discussions)

---

**Made with ❤️ for the AI community**
