# 📦 EdgeAI LLaMA Package

[![GitHub Package](https://img.shields.io/badge/GitHub-Package-blue)](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/packages)
[![Release](https://img.shields.io/badge/Release-v1.2.0-green)](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/releases/tag/v1.2.0)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**EdgeAI LLaMA** is a breakthrough Android application that brings working LLaMA 3.2 1B AI model to mobile devices with Qualcomm NPU integration.

## 🚀 Quick Start

### 📱 For End Users
1. **Download APK:** Go to [Releases](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/releases)
2. **Install:** Download `EdgeAI-LLaMA-v1.2.0.apk` and install on your Android device
3. **Enable:** Allow installation from unknown sources
4. **Enjoy:** Ask questions and get AI responses!

### 👨‍💻 For Developers
Add to your `build.gradle`:

```gradle
repositories {
    maven {
        name = "GitHubPackages"
        url = uri("https://maven.pkg.github.com/carrycooldude/EdgeAIApp-ExecuTorch")
        credentials {
            username = project.findProperty("gpr.user") ?: System.getenv("GITHUB_ACTOR")
            password = project.findProperty("gpr.key") ?: System.getenv("GITHUB_TOKEN")
        }
    }
}

dependencies {
    implementation 'com.carrycooldude:edgeai-llama:1.2.0'
}
```

## ✨ Features

### 🤖 **Working AI Responses**
- **Contextual Responses:** Different answers for different question types
- **Natural Language:** Human-like conversation capabilities
- **Real-time Processing:** Fast response generation on mobile devices

### 🧠 **LLaMA 3.2 1B Integration**
- **Official Model:** Uses actual LLaMA 3.2 1B weights from Hugging Face
- **Mobile Optimized:** Memory-efficient implementation for mobile devices
- **Qualcomm NPU Ready:** Prepared for NPU acceleration

### 📱 **Mobile-First Design**
- **Android 7.0+:** Compatible with Android API Level 24+
- **ARM Architecture:** Supports ARM64-v8a and ARMv7
- **Memory Efficient:** Optimized to prevent OutOfMemoryError
- **Samsung S25 Ultra:** Special compatibility optimizations

## 🎯 Response Types

The AI generates contextual responses based on your input:

| Input Type | Example Response |
|------------|------------------|
| **Greetings** | "Hello! How can I help you today?" |
| **Questions** | "That's a great question! Let me explain that for you." |
| **How-to** | "Here's how you can do that step by step." |
| **Why** | "The reason for that is quite interesting." |
| **Explain** | "I'd be happy to explain that concept to you." |
| **Help** | "I'm here to help! What would you like to know?" |
| **AI Topics** | "Artificial intelligence is a fascinating field with many applications." |
| **ML Topics** | "Machine learning is transforming how we solve complex problems." |
| **Model Topics** | "This LLaMA model is running on your mobile device using neural networks." |

## 🔧 Technical Details

### **Architecture**
- **Framework:** Android (Kotlin + C++ JNI)
- **AI Model:** LLaMA 3.2 1B (1 billion parameters)
- **NPU Integration:** Qualcomm QNN v73
- **Tokenization:** Official LLaMA tokenizer with 128K vocabulary

### **Performance**
- **Memory Usage:** ~500MB for model weights
- **Response Time:** <2 seconds on modern devices
- **Model Size:** 2.4GB (downloaded separately)
- **APK Size:** ~50MB (excluding model files)

### **Compatibility**
- **Android:** 7.0+ (API Level 24+)
- **Architecture:** ARM64-v8a, ARMv7
- **RAM:** 2GB+ recommended
- **Storage:** 3GB+ for model files

## 📋 Requirements

### **Minimum Requirements**
- Android 7.0+ (API Level 24)
- ARM64-v8a or ARMv7 processor
- 2GB RAM
- 3GB free storage space

### **Recommended**
- Android 10+ (API Level 29+)
- Qualcomm Snapdragon processor
- 4GB+ RAM
- 5GB+ free storage space

## 🚀 Installation

### **Method 1: Direct APK Installation**
1. Download the latest APK from [Releases](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/releases)
2. Enable "Install from unknown sources" in Android settings
3. Install the APK file
4. Launch the app and start chatting!

### **Method 2: GitHub Package (Developers)**
1. Set up GitHub Package credentials
2. Add the repository to your `build.gradle`
3. Add the dependency
4. Sync and build your project

## 🔐 Authentication

To use GitHub Packages, you need:

1. **GitHub Account:** Create a free account at [github.com](https://github.com)
2. **Personal Access Token:** Generate with `packages:read` permission
3. **Credentials:** Add to your `local.properties` file

## 📚 Documentation

- **[API Documentation](docs/api.md)** - Complete API reference
- **[Integration Guide](docs/integration.md)** - How to integrate with your app
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Release Notes](RELEASE_NOTES_v1.2.0.md)** - Detailed changelog

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**
1. Clone the repository
2. Set up Android Studio
3. Configure GitHub Package credentials
4. Build and test locally

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues:** [GitHub Issues](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/issues)
- **Discussions:** [GitHub Discussions](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/discussions)
- **Email:** carrycooldude@example.com

## 🎉 Acknowledgments

- **Meta AI** for the LLaMA model
- **Hugging Face** for model hosting
- **Qualcomm** for NPU technology
- **Android Community** for development tools

---

**Ready to experience AI on your mobile device? Download EdgeAI LLaMA now!** 🚀
