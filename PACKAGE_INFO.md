# EdgeAI LLaMA Package Configuration

## Package Information
- **Package Name:** `com.carrycooldude:edgeai-llama`
- **Version:** `1.2.0`
- **Repository:** GitHub Packages
- **Package Type:** Android APK + Maven Package

## Installation Instructions

### For End Users (APK Installation)
1. Go to the [Releases page](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/releases)
2. Download the latest APK file (`EdgeAI-LLaMA-v1.2.0.apk`)
3. Enable "Install from unknown sources" on your Android device
4. Install the APK and enjoy working AI responses!

### For Developers (Maven Package)
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

## Package Contents
- **APK:** Complete Android application with working LLaMA model
- **Sources:** Source code for integration and customization
- **Documentation:** Complete API documentation and usage examples
- **Models:** Pre-trained LLaMA 3.2 1B model files
- **Libraries:** Qualcomm QNN integration libraries

## Features
- ✅ Working LLaMA model with actual AI responses
- ✅ Contextual responses for different question types
- ✅ Qualcomm NPU integration ready
- ✅ Mobile-optimized implementation
- ✅ Memory-efficient design
- ✅ Samsung S25 Ultra compatibility

## Requirements
- Android API Level 24+ (Android 7.0+)
- ARM64-v8a or ARMv7 architecture
- 2GB+ RAM recommended
- Qualcomm Snapdragon processor (for NPU acceleration)

## Support
- **Issues:** [GitHub Issues](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/issues)
- **Discussions:** [GitHub Discussions](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/discussions)
- **Documentation:** [README.md](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/blob/main/README.md)

## License
MIT License - see [LICENSE](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/blob/main/LICENSE) file for details.
