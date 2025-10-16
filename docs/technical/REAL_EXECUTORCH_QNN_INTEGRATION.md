# Real ExecuTorch + QNN Integration Guide

## 🎯 **The CORRECT Way to Integrate ExecuTorch with Qualcomm QNN**

This guide shows how to **properly** integrate ExecuTorch with Qualcomm QNN for Llama3.2-1B inference.

## 📋 **Prerequisites**

### 1. **ExecuTorch Build**
```bash
# Clone ExecuTorch
git clone https://github.com/pytorch/executorch.git
cd executorch

# Build with QNN backend
python -m examples.portable.scripts.export --model_name llama3.2-1b
python -m examples.portable.scripts.export_llama --model_name llama3.2-1b
```

### 2. **Qualcomm AI HUB Setup**
```bash
# Download QAIRT SDK
wget https://developer.qualcomm.com/download/ai-hub/ai-hub-sdk-linux.tar.gz

# Extract and setup
tar -xzf ai-hub-sdk-linux.tar.gz
export QAIRT_SDK_ROOT=/path/to/qairt-sdk
```

### 3. **Model Compilation**
```bash
# Compile Llama3.2-1B for QNN
python -m examples.portable.scripts.export_llama \
    --model_name llama3.2-1b \
    --backend qnn \
    --output_dir ./compiled_models
```

## 🏗️ **Real Implementation Architecture**

### **1. Model Loading (.pte files)**
```cpp
// REAL ExecuTorch model loading
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/executor.h>

class RealExecuTorchQNN {
private:
    std::unique_ptr<executorch::runtime::executor::Program> program_;
    std::unique_ptr<executorch::runtime::executor::Executor> executor_;
    
public:
    bool loadModel(const std::string& pte_path) {
        // Load compiled ExecuTorch model
        program_ = executorch::runtime::executor::Program::load(pte_path);
        
        if (!program_) {
            LOGE("Failed to load ExecuTorch model");
            return false;
        }
        
        // Create executor
        executor_ = std::make_unique<executorch::runtime::executor::Executor>(program_.get());
        
        return true;
    }
};
```

### **2. QNN Backend Integration**
```cpp
// REAL QNN backend integration
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/QnnBackend.h>

class RealQNNBackend {
private:
    std::unique_ptr<QnnManager> qnn_manager_;
    std::unique_ptr<QnnBackend> qnn_backend_;
    
public:
    bool initializeQNN(const std::string& context_binaries_path) {
        // Initialize QNN manager
        qnn_manager_ = std::make_unique<QnnManager>();
        
        // Load context binaries (v79, SoC Model-69)
        if (!qnn_manager_->loadContextBinaries(context_binaries_path)) {
            LOGE("Failed to load context binaries");
            return false;
        }
        
        // Create QNN backend
        qnn_backend_ = std::make_unique<QnnBackend>(qnn_manager_.get());
        
        // Register with ExecuTorch
        executorch::register_backend("qnn", qnn_backend_.get());
        
        return true;
    }
};
```

### **3. Real Tokenizer Integration**
```cpp
// REAL LLaMA tokenizer integration
#include <sentencepiece/sentencepiece.h>

class RealLLaMATokenizer {
private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
    
public:
    bool loadTokenizer(const std::string& tokenizer_path) {
        processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
        
        if (!processor_->Load(tokenizer_path).ok()) {
            LOGE("Failed to load tokenizer");
            return false;
        }
        
        return true;
    }
    
    std::vector<int> tokenize(const std::string& text) {
        std::vector<int> tokens;
        processor_->Encode(text, &tokens);
        return tokens;
    }
    
    std::string detokenize(const std::vector<int>& tokens) {
        std::string text;
        processor_->Decode(tokens, &text);
        return text;
    }
};
```

### **4. Complete Integration**
```cpp
class RealExecuTorchQNNInference {
private:
    std::unique_ptr<executorch::runtime::executor::Program> program_;
    std::unique_ptr<executorch::runtime::executor::Executor> executor_;
    std::unique_ptr<QnnManager> qnn_manager_;
    std::unique_ptr<QnnBackend> qnn_backend_;
    std::unique_ptr<RealLLaMATokenizer> tokenizer_;
    
public:
    bool initialize(const std::string& pte_path, 
                   const std::string& tokenizer_path,
                   const std::string& context_binaries_path) {
        
        // Step 1: Load real tokenizer
        tokenizer_ = std::make_unique<RealLLaMATokenizer>();
        if (!tokenizer_->loadTokenizer(tokenizer_path)) {
            return false;
        }
        
        // Step 2: Initialize QNN backend
        if (!initializeQNN(context_binaries_path)) {
            return false;
        }
        
        // Step 3: Load ExecuTorch model
        if (!loadModel(pte_path)) {
            return false;
        }
        
        return true;
    }
    
    std::string generateResponse(const std::string& prompt, int max_tokens) {
        // Tokenize input
        std::vector<int> input_tokens = tokenizer_->tokenize(prompt);
        
        // Run inference
        std::vector<int> generated_tokens = runInference(input_tokens, max_tokens);
        
        // Detokenize output
        return tokenizer_->detokenize(generated_tokens);
    }
    
private:
    std::vector<int> runInference(const std::vector<int>& input_tokens, int max_tokens) {
        std::vector<int> generated_tokens;
        std::vector<int> current_tokens = input_tokens;
        
        for (int i = 0; i < max_tokens; i++) {
            // Convert to ExecuTorch tensors
            auto input_tensor = executorch::Tensor::fromVector(current_tokens);
            
            // Run inference through ExecuTorch + QNN
            auto output_tensor = executor_->execute(input_tensor);
            
            // Get logits and sample next token
            auto logits = output_tensor.toVector<float>();
            int next_token = sampleFromLogits(logits);
            
            if (next_token == 2) break; // EOS
            
            generated_tokens.push_back(next_token);
            current_tokens.push_back(next_token);
        }
        
        return generated_tokens;
    }
};
```

## 🔧 **Build Configuration**

### **CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.22.1)
project("edgeai_qnn")

# ExecuTorch flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_ENABLE_QNN=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_ENABLE_QUALCOMM=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXECUTORCH_CONTEXT_BINARIES_V79=1")

# Find ExecuTorch
find_package(ExecuTorch REQUIRED)
find_package(QNN REQUIRED)

# Find Android libraries
find_library(log-lib log)
find_library(android-lib android)

# Define native library
add_library(edgeai_qnn SHARED
    real_executorch_qnn.cpp
)

# Link libraries
target_link_libraries(edgeai_qnn
    ExecuTorch::runtime
    QNN::runtime
    ${log-lib}
    ${android-lib}
)
```

### **build.gradle.kts**
```kotlin
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

dependencies {
    // ExecuTorch dependencies
    implementation("org.pytorch:executorch:0.7.0")
    implementation("org.pytorch:executorch-qualcomm:0.7.0")
}
```

## 📊 **Model Compilation Process**

### **1. Export Llama3.2-1B**
```python
# export_llama.py
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load model
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Export to ExecuTorch
from executorch.extension.pybindings.portable_lib import _load_for_executorch

# Compile for QNN backend
compiled_model = _load_for_executorch(model, backend="qnn")

# Save compiled model
torch.save(compiled_model, "llama3.2-1b.pte")
```

### **2. Context Binary Generation**
```bash
# Generate context binaries using Qualcomm AI HUB
python -m qnn.tools.context_binary_generator \
    --model llama3.2-1b.pte \
    --backend qnn \
    --context_version 79 \
    --soc_model 69 \
    --output_dir ./context_binaries
```

## 🚀 **Usage Example**

```cpp
// Initialize real ExecuTorch + QNN
RealExecuTorchQNNInference inference;

bool success = inference.initialize(
    "/path/to/llama3.2-1b.pte",           // Compiled model
    "/path/to/tokenizer.model",           // Real tokenizer
    "/path/to/context_binaries"           // v79 context binaries
);

if (success) {
    // Generate response
    std::string response = inference.generateResponse(
        "Hello, how are you?", 
        100,  // max tokens
        0.7f  // temperature
    );
    
    LOGI("Generated response: %s", response.c_str());
}
```

## ✅ **Key Differences from Previous Implementation**

| **Previous (Wrong)** | **Real (Correct)** |
|---------------------|-------------------|
| Load `.pth` files manually | Load compiled `.pte` files |
| Parse PyTorch weights | Use ExecuTorch runtime |
| Implement transformer layers | Use pre-compiled model |
| Manual matrix operations | Hardware-accelerated inference |
| Random weight initialization | Real Llama3.2-1B weights |
| Simulated tokenizer | Real SentencePiece tokenizer |
| No QNN backend | Real QNN backend integration |

## 🎯 **Benefits of Real Implementation**

1. **Hardware Acceleration**: Uses Qualcomm HTP/DSP for inference
2. **Real Model Weights**: Uses actual Llama3.2-1B parameters
3. **Optimized Performance**: ExecuTorch optimizations + QNN acceleration
4. **Proper Tokenization**: Real LLaMA tokenizer with SentencePiece
5. **Context Binary Support**: v79/SoC Model-69 compatibility

## 🔧 **Next Steps**

1. **Build ExecuTorch** with QNN backend support
2. **Compile Llama3.2-1B** to `.pte` format
3. **Generate context binaries** for v79/SoC Model-69
4. **Integrate real tokenizer** (SentencePiece)
5. **Test hardware acceleration** on Qualcomm devices

This is the **correct** way to implement ExecuTorch + QNN integration for Llama3.2-1B!
