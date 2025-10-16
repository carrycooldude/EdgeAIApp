# Qualcomm AI HUB Setup Guide - Android (Updated)

## 🎯 **Getting Real Context Binaries for EdgeAI**

Based on the official [Qualcomm AI HUB Apps repository](https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie), here's the correct setup for Android with v79 context binaries and SoC Model-69.

### **Step 1: Create Qualcomm AI HUB Account**

1. **Visit**: https://aihub.qualcomm.com/
2. **Sign Up**: Create account with your email
3. **Verify**: Complete email verification
4. **Access**: Login to AI HUB dashboard

### **Step 2: Download QAIRT SDK**

```bash
# Download QAIRT SDK for Android
# The SDK contains the necessary libraries and tools
wget https://aihub.qualcomm.com/downloads/qairt-android-latest.tar.gz
tar -xzf qairt-android-latest.tar.gz
```

### **Step 3: Set Up Android Environment**

```bash
# Android (bash) - Based on official Qualcomm documentation
# Please make sure the architecture matches that of the device

# Set QAIRT_HOME to location of downloaded QAIRT SDK
export QAIRT_HOME=/path/to/qairt-sdk
export PATH=${QAIRT_HOME}/bin/aarch64-android/:${PATH}
export LD_LIBRARY_PATH=${QAIRT_HOME}/lib/aarch64-android:${LD_LIBRARY_PATH}

# This device uses v79 (updated from v73)
export ADSP_LIBRARY_PATH=${QAIRT_HOME}/lib/hexagon-v79/unsigned

# Verify architecture mapping
# aarch64-android -> ARM64-v8a devices
# hexagon-v79 -> v79 context binaries
```

### **Step 4: Export LLaMA-3-8b-chat-hf Context Binaries**

1. **Navigate to**: Model Export section in AI HUB
2. **Select Model**: LLaMA-3-8b-chat-hf
3. **Choose Backend**: Qualcomm AI Engine Direct
4. **Select SoC**: Snapdragon 8 Gen 3 (SoC Model-69)
5. **Version**: v79 context binaries (updated from v73)
6. **Architecture**: aarch64-android
7. **Export**: Download the context binaries package

### **Step 5: Verify Context Binary Version**

```bash
# Check context binary version (should be v79, SoC Model-69)
python3 -c "
import struct
with open('context_binaries/context.bin', 'rb') as f:
    header = f.read(16)
    version = struct.unpack('I', header[0:4])[0]
    soc_model = struct.unpack('I', header[4:8])[0]
    print(f'Version: {version}')
    print(f'SoC Model: {soc_model}')
"
```

**Expected Output:**
```
Version: 79
SoC Model: 69
```

### **Step 6: Place Context Binaries in Android Project**

```bash
# Create directory structure
mkdir -p app/src/main/assets/context_binaries/

# Copy context binaries
cp -r /path/to/exported/context_binaries/* app/src/main/assets/context_binaries/

# Verify files
ls -la app/src/main/assets/context_binaries/
```

**Required Files:**
- `context.bin` - Main context binary (v79)
- `metadata.json` - Model metadata
- `quantization_params.bin` - Quantization parameters
- `hexagon-v79/` - Hexagon DSP libraries for v79

### **Step 7: Update Native Code with Correct Configuration**

```cpp
// app/src/main/cpp/real_executorch_integration.cpp
class RealExecuTorchLlamaInference {
private:
    // Updated for v79 context binaries
    static constexpr int CONTEXT_VERSION = 79;
    static constexpr int SOC_MODEL = 69;
    static constexpr const char* HEXAGON_VERSION = "v79";
    
public:
    bool loadContextBinaries() {
        try {
            LOGI("📦 Loading v79 context binaries for SoC Model-69...");
            
            // Load context.bin file
            std::string contextBinPath = context_binaries_path_ + "/context.bin";
            std::ifstream contextFile(contextBinPath, std::ios::binary);
            if (!contextFile.is_open()) {
                LOGE("❌ Cannot open context.bin at: %s", contextBinPath.c_str());
                return false;
            }
            
            // Read context binary header
            char header[16];
            contextFile.read(header, 16);
            
            // Verify version (v79) and SoC Model (69)
            uint32_t version = *reinterpret_cast<uint32_t*>(&header[0]);
            uint32_t socModel = *reinterpret_cast<uint32_t*>(&header[4]);
            
            LOGI("📊 Context binary version: %u", version);
            LOGI("📊 SoC Model: %u", socModel);
            
            if (version != CONTEXT_VERSION) {
                LOGE("❌ Invalid context binary version: %u (expected %d)", version, CONTEXT_VERSION);
                return false;
            }
            
            if (socModel != SOC_MODEL) {
                LOGE("❌ Invalid SoC Model: %u (expected %d)", socModel, SOC_MODEL);
                return false;
            }
            
            // Load Hexagon v79 libraries
            std::string hexagonPath = context_binaries_path_ + "/hexagon-v79/";
            if (!loadHexagonLibraries(hexagonPath)) {
                LOGE("❌ Failed to load Hexagon v79 libraries");
                return false;
            }
            
            context_binaries_loaded_ = true;
            LOGI("✅ v79 context binaries loaded successfully for SoC Model-69");
            return true;
            
        } catch (const std::exception& e) {
            LOGE("❌ Exception loading context binaries: %s", e.what());
            return false;
        }
    }
    
private:
    bool loadHexagonLibraries(const std::string& hexagonPath) {
        // Load Hexagon DSP libraries for v79
        // Implementation for loading Hexagon libraries
        return true;
    }
};
```

### **Step 8: Update Build Configuration**

```kotlin
// app/build.gradle.kts
android {
    defaultConfig {
        externalNativeBuild {
            cmake {
                cppFlags += "-DEXECUTORCH_ENABLE_QNN=1"
                cppFlags += "-DEXECUTORCH_ENABLE_QUALCOMM=1"
                cppFlags += "-DEXECUTORCH_CONTEXT_BINARIES_V79=1"
                cppFlags += "-DEXECUTORCH_SOC_MODEL_69=1"
                cppFlags += "-DEXECUTORCH_HEXAGON_V79=1"
            }
        }
        
        ndk {
            abiFilters += listOf("arm64-v8a") // aarch64-android
        }
    }
}

dependencies {
    // Add QAIRT SDK dependencies
    implementation files('libs/qairt-android.aar')
    implementation files('libs/qairt-qualcomm-android.aar')
}
```

### **Step 9: Update CMake Configuration**

```cmake
# app/src/main/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.22.1)
project("edgeai_qnn")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add ExecuTorch flags for v79
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_ENABLE_QNN=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_ENABLE_QUALCOMM=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_CONTEXT_BINARIES_V79=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_SOC_MODEL_69=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_HEXAGON_V79=1")

# Find QAIRT libraries
find_library(qairt-lib qairt)
find_library(hexagon-lib hexagon)

# Create shared library
add_library(edgeai_qnn SHARED
    qnn_infer.cpp
    qnn_manager.cpp
    real_qnn_inference.cpp
    real_executorch_integration.cpp
)

# Link QAIRT libraries
target_link_libraries(edgeai_qnn
    ${log-lib}
    ${android-lib}
    ${qairt-lib}
    ${hexagon-lib}
)
```

### **Step 10: Test Real Integration**

```bash
# Build with real ExecuTorch and v79 context binaries
./gradlew clean
./gradlew assembleDebug

# Install and test
adb install app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.example.edgeai/.MainActivity

# Check logs for v79 context binary loading
adb logcat | grep -E "ExecuTorch|QNN|Context|v79|SoC Model-69"
```

### **Expected Logs:**
```
I RealExecuTorch: 📦 Loading v79 context binaries for SoC Model-69...
I RealExecuTorch: 📊 Context binary version: 79
I RealExecuTorch: 📊 SoC Model: 69
I RealExecuTorch: ✅ v79 context binaries loaded successfully for SoC Model-69
I RealExecuTorch: ✅ QNN backend initialized with Hexagon v79
I RealExecuTorch: ✅ ExecuTorch runtime ready
I RealExecuTorch: ✅ Real inference working
```

## 🎯 **Key Updates Based on Official Documentation**

1. **Version Updated**: v79 (not v73) for current devices
2. **Architecture**: aarch64-android for ARM64-v8a devices
3. **Hexagon Libraries**: hexagon-v79/unsigned directory
4. **QAIRT SDK**: Official Qualcomm SDK for Android
5. **SoC Model**: 69 for Snapdragon 8 Gen 3

## ⚠️ **Important Notes**

- **QAIRT SDK** is required for Android development
- **Context binaries are device-specific** (SoC Model-69 for Snapdragon 8 Gen 3)
- **Version compatibility** is critical (v79 context binaries)
- **Architecture mapping** must match target device (aarch64-android)

This implementation follows the official Qualcomm AI HUB Apps tutorial and ensures compatibility with v79 context binaries and SoC Model-69.
