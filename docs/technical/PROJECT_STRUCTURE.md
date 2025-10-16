# Project Structure Documentation

## 📁 **EdgeAI Project Organization**

This document describes the improved project structure for EdgeAI v1.3.0 with real ExecuTorch + QNN integration.

## 🏗️ **Directory Structure**

```
EdgeAI/
├── 📱 app/                                    # Android application
│   ├── src/main/
│   │   ├── 📱 cpp/                           # Native C++ implementation
│   │   │   ├── real_executorch_qnn.cpp      # 🎯 Main ExecuTorch + QNN integration
│   │   │   ├── CMakeLists.txt               # Build configuration
│   │   │   ├── qnn_infer.cpp                # QNN inference utilities
│   │   │   ├── qnn_manager.cpp              # QNN manager implementation
│   │   │   └── ...                          # Other C++ files
│   │   ├── ☕ java/                          # Kotlin/Java code
│   │   │   └── com/example/edgeai/
│   │   │       ├── ml/                      # Machine learning classes
│   │   │       │   └── LLaMAInference.kt   # 🎯 Main inference class
│   │   │       ├── ui/                      # User interface
│   │   │       └── ...                      # Other classes
│   │   ├── 📦 assets/                       # Model files and resources
│   │   │   ├── models/                      # Model files
│   │   │   │   ├── Llama3.2-1B/            # Llama3.2-1B model
│   │   │   │   └── ...                      # Other models
│   │   │   ├── context_binaries/            # QNN context binaries
│   │   │   └── tokenizer/                   # Tokenizer files
│   │   └── 📱 res/                          # Android resources
│   └── build.gradle.kts                     # App build configuration
├── 📚 docs/                                  # Documentation
│   ├── technical/                           # Technical documentation
│   │   ├── REAL_EXECUTORCH_QNN_INTEGRATION.md  # 🎯 Integration guide
│   │   ├── IMPLEMENTATION_ANALYSIS.md         # Implementation analysis
│   │   └── ARCHITECTURE.md                    # Architecture overview
│   ├── setup/                               # Setup guides
│   │   ├── QUALCOMM_AIHUB_SETUP.md          # Qualcomm AI HUB setup
│   │   ├── EXECUTORCH_SETUP.md              # ExecuTorch setup
│   │   └── ANDROID_SETUP.md                 # Android setup
│   └── releases/                            # Release notes
│       ├── RELEASE_NOTES_v1.3.0.md          # Latest release
│       ├── RELEASE_NOTES_v1.2.0.md          # Previous releases
│       └── ...
├── 🔧 scripts/                              # Build and setup scripts
│   ├── setup_real_executorch.ps1            # 🎯 Main setup script
│   ├── copy_model_to_device.ps1             # Model deployment
│   └── ...                                  # Other scripts
├── 📦 external_models/                      # External model files
│   └── Llama-3-8b-chat-hf/                 # Large model files
├── 🏗️ build.gradle.kts                      # Project build configuration
├── 📋 settings.gradle.kts                   # Project settings
├── 📄 README.md                             # 🎯 Main project documentation
├── 📄 LICENSE                                # License file
└── 📄 CONTRIBUTING.md                       # Contributing guidelines
```

## 🎯 **Key Files and Their Purpose**

### **Core Implementation Files**

| File | Purpose | Importance |
|------|---------|------------|
| `app/src/main/cpp/real_executorch_qnn.cpp` | **Main ExecuTorch + QNN integration** | 🎯 **Critical** |
| `app/src/main/java/com/example/edgeai/ml/LLaMAInference.kt` | **Main inference class** | 🎯 **Critical** |
| `app/src/main/cpp/CMakeLists.txt` | **Native build configuration** | 🔧 **Important** |
| `app/build.gradle.kts` | **Android build configuration** | 🔧 **Important** |

### **Documentation Files**

| File | Purpose | Importance |
|------|---------|------------|
| `README.md` | **Main project documentation** | 🎯 **Critical** |
| `docs/technical/REAL_EXECUTORCH_QNN_INTEGRATION.md` | **Integration guide** | 🎯 **Critical** |
| `docs/technical/IMPLEMENTATION_ANALYSIS.md` | **Implementation analysis** | 📖 **Important** |
| `docs/setup/QUALCOMM_AIHUB_SETUP.md` | **Setup guide** | 📖 **Important** |

### **Script Files**

| File | Purpose | Importance |
|------|---------|------------|
| `scripts/setup_real_executorch.ps1` | **Main setup script** | 🔧 **Important** |
| `scripts/copy_model_to_device.ps1` | **Model deployment** | 🔧 **Important** |

## 🔧 **Build System**

### **Android Build (Gradle)**

```kotlin
// app/build.gradle.kts
android {
    defaultConfig {
        ndk {
            abiFilters += listOf("arm64-v8a") // For v79 context binaries
        }
    }
    
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}
```

### **Native Build (CMake)**

```cmake
# app/src/main/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.22.1)
project("edgeai_qnn")

# ExecuTorch flags for v79
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_ENABLE_QNN=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_CONTEXT_BINARIES_V79=1")

# Define native library
add_library(edgeai_qnn SHARED
    real_executorch_qnn.cpp  # Main implementation
    qnn_infer.cpp
    qnn_manager.cpp
)
```

## 📱 **Android Architecture**

### **UI Layer**
- **MainActivity.kt**: Main activity with UI
- **LLaMAInference.kt**: Core inference logic
- **JNI Bridge**: Communication between Kotlin and C++

### **Native Layer**
- **real_executorch_qnn.cpp**: ExecuTorch + QNN integration
- **qnn_infer.cpp**: QNN inference utilities
- **qnn_manager.cpp**: QNN manager implementation

### **Asset Layer**
- **models/**: Model files (.pte, .pth)
- **context_binaries/**: QNN context binaries
- **tokenizer/**: Tokenizer files

## 🚀 **Development Workflow**

### **1. Setup**
```bash
# Run setup script
.\scripts\setup_real_executorch.ps1
```

### **2. Development**
```bash
# Build debug version
.\gradlew assembleDebug

# Install on device
adb install app\build\outputs\apk\debug\app-debug.apk
```

### **3. Testing**
```bash
# Run tests
.\gradlew test

# Run Android tests
.\gradlew connectedAndroidTest
```

### **4. Release**
```bash
# Build release version
.\gradlew assembleRelease

# Generate release APK
.\gradlew bundleRelease
```

## 📊 **File Size Analysis**

### **Large Files**
- `external_models/Llama-3-8b-chat-hf/consolidated.00.pth`: ~2.3GB
- `app/build/outputs/apk/debug/app-debug.apk`: ~50MB
- `app/build/outputs/apk/release/app-release.apk`: ~30MB

### **Documentation**
- `README.md`: ~15KB
- `docs/technical/REAL_EXECUTORCH_QNN_INTEGRATION.md`: ~25KB
- `docs/technical/IMPLEMENTATION_ANALYSIS.md`: ~10KB

## 🔍 **Code Organization**

### **C++ Code Structure**
```cpp
// real_executorch_qnn.cpp
class RealExecuTorchQNNInference {
private:
    // Model configuration
    // ExecuTorch components
    // QNN backend components
    // Tokenizer components
    
public:
    // Initialization
    // Inference methods
    // Utility methods
};
```

### **Kotlin Code Structure**
```kotlin
// LLaMAInference.kt
class LLaMAInference {
    companion object {
        // Native method declarations
        // Constants
    }
    
    // Initialization methods
    // Inference methods
    // Utility methods
}
```

## 🎯 **Best Practices**

### **File Naming**
- Use descriptive names: `real_executorch_qnn.cpp`
- Include version info: `RELEASE_NOTES_v1.3.0.md`
- Use consistent casing: `LLaMAInference.kt`

### **Documentation**
- Keep README.md up to date
- Document all public APIs
- Include code examples
- Use clear, concise language

### **Code Organization**
- Separate concerns into different files
- Use meaningful class and method names
- Add comments for complex logic
- Follow consistent formatting

## 🔧 **Maintenance**

### **Regular Tasks**
- Update documentation with new features
- Clean up unused files
- Optimize build configuration
- Update dependencies

### **Version Management**
- Update version numbers in build files
- Create release notes for each version
- Tag releases in Git
- Update documentation

This improved structure makes the project more maintainable, easier to understand, and better organized for future development.
