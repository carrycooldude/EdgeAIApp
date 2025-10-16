# EdgeAI Documentation

Welcome to the EdgeAI documentation website! This site provides comprehensive documentation for the EdgeAI project - Real ExecuTorch + QNN Integration for Llama3.2-1B inference on Android.

## 🚀 Quick Navigation

- **[Overview](#overview)** - What is EdgeAI?
- **[Features](#features)** - Key capabilities and benefits
- **[Setup](#setup)** - Getting started guide
- **[Documentation](#documentation)** - Technical documentation
- **[Releases](#releases)** - Latest releases and updates

## 📚 Documentation Structure

### Technical Documentation
- [Real ExecuTorch + QNN Integration](technical/REAL_EXECUTORCH_QNN_INTEGRATION.md)
- [Implementation Analysis](technical/IMPLEMENTATION_ANALYSIS.md)
- [Project Structure](technical/PROJECT_STRUCTURE.md)

### Setup Guides
- [Qualcomm AI HUB Setup](setup/QUALCOMM_AIHUB_SETUP.md)
- [ExecuTorch Configuration](setup/EXECUTORCH_SETUP.md)
- [Android Development Setup](setup/ANDROID_SETUP.md)

### Release Notes
- [v1.4.0 Release Notes](releases/RELEASE_NOTES_v1.4.0.md)
- [v1.3.0 Release Notes](releases/RELEASE_NOTES_v1.3.0.md)
- [v1.2.0 Release Notes](releases/RELEASE_NOTES_v1.2.0.md)

## 🎯 What's New in v1.4.0

- ✅ **Real ExecuTorch Integration**: Proper .pte model loading instead of manual implementation
- ✅ **QNN Backend Support**: Hardware acceleration with v79 context binaries
- ✅ **Actual Llama3.2-1B**: Uses real model weights, not simulated responses
- ✅ **Improved Architecture**: Clean separation of concerns and proper documentation
- ✅ **Better Performance**: Optimized inference pipeline with hardware acceleration

## 🔧 Quick Start

```bash
# Clone repository
git clone https://github.com/carrycooldude/EdgeAIApp-ExecuTorch.git
cd EdgeAIApp-ExecuTorch

# Run setup script
.\scripts\setup_real_executorch.ps1

# Build and install
.\gradlew assembleDebug
adb install app\build\outputs\apk\debug\app-debug.apk
```

## 📞 Support

- 📧 Email: carrycooldude@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/discussions)

---

**Made with ❤️ for the AI community**
