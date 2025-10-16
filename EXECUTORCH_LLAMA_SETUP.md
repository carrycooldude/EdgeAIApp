# ExecuTorch Llama-3-8b-chat-hf Integration

This project demonstrates how to run Llama-3-8b-chat-hf on Android devices using ExecuTorch with Qualcomm HTP backend. The model is precompiled into context binaries by Qualcomm AI HUB for optimal performance on Qualcomm SoCs.

## Overview

- **Model**: Llama-3-8b-chat-hf (8 billion parameters)
- **Backend**: ExecuTorch with Qualcomm HTP acceleration
- **Platform**: Android (API level 24+)
- **Hardware**: Qualcomm SoCs with AI Engine Direct support
- **Memory**: 16GB RAM recommended
- **Context Binary Version**: v79 (SoC Model-69)

## Prerequisites

### 1. ExecuTorch Setup
Follow the [ExecuTorch installation guide](https://github.com/pytorch/executorch) to set up ExecuTorch.

### 2. Qualcomm AI Engine Direct Backend
Build the Qualcomm AI Engine Direct Backend following the [official tutorial](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html).

### 3. Qualcomm AI HUB Account
- Create an account at [Qualcomm AI HUB](https://aihub.qualcomm.com/)
- Get access to export context binaries for Llama-3-8b-chat-hf

## Setup Instructions

### Step 1: Install Dependencies

```bash
# Install ExecuTorch and related packages
pip3 install executorch torch transformers optimum-executorch

# Run the setup script
chmod +x setup_executorch_llama.sh
./setup_executorch_llama.sh
```

### Step 2: Prepare Model Files

#### Export Context Binaries
1. Follow instructions at [Qualcomm Llama-v3-8B-Chat](https://huggingface.co/qualcomm/Llama-v3-8B-Chat)
2. Export context binaries using Qualcomm AI HUB
3. Place the exported binaries in `app/src/main/assets/context_binaries/`

#### Download Tokenizer
1. Visit [Meta Llama Models](https://github.com/meta-llama/llama-models/blob/main/README.md)
2. Download `tokenizer.model` file
3. Place it in `app/src/main/assets/tokenizer/`

### Step 3: Verify Context Binary Version

Ensure your context binaries are:
- **Version**: v79
- **Compatible with**: SoC Model-69
- **Required files**:
  - `libQnnHtp.so`
  - `libQnnHtpV79Stub.so`
  - `libQnnSystem.so`

### Step 4: Build and Run

```bash
# Build the project
./gradlew assembleDebug

# Install on device
./gradlew installDebug

# Test with sample prompt
# Use the app to test with: "What is baseball?"
```

## Usage

### Kotlin Integration

```kotlin
// Initialize ExecuTorch Llama-3-8b-chat-hf
val llamaInference = LLaMAInference(context)

// Initialize with model paths
val success = llamaInference.initializeExecuTorchLlama(
    modelPath = "path/to/model",
    tokenizerPath = "path/to/tokenizer.model",
    contextBinariesPath = "path/to/context_binaries"
)

if (success) {
    // Generate response
    val response = llamaInference.generateExecuTorchLlama(
        prompt = "What is baseball?",
        maxTokens = 128,
        temperature = 0.8f
    )
    println("Response: $response")
}
```

### Native C++ Integration

The native implementation provides:
- ExecuTorch runtime initialization
- Qualcomm QNN backend integration
- Context binary loading and verification
- Token generation with temperature sampling

## Project Structure

```
app/src/main/
├── cpp/
│   ├── executorch_llama.cpp      # ExecuTorch Llama integration
│   ├── qnn_infer.cpp             # QNN inference utilities
│   ├── qnn_manager.cpp           # QNN manager
│   └── real_qnn_inference.cpp    # Real QNN inference
├── java/com/example/edgeai/ml/
│   └── LLaMAInference.kt         # Kotlin wrapper
└── assets/
    ├── models/Llama-3-8b-chat-hf/    # Model files
    ├── tokenizer/                     # Tokenizer files
    └── context_binaries/              # Qualcomm context binaries
```

## Key Features

### ExecuTorch Integration
- **Native PyTorch Export**: Direct export from PyTorch without intermediate conversions
- **Production-Proven**: Powers billions of users at Meta
- **Tiny Runtime**: 50KB base footprint
- **Hardware Backends**: 12+ hardware backends including Qualcomm

### Qualcomm HTP Backend
- **AI Engine Direct**: Leverages Qualcomm's dedicated AI hardware
- **Context Binaries**: Pre-compiled for optimal performance
- **Version Compatibility**: v79 for SoC Model-69
- **Memory Optimization**: Efficient memory usage for mobile devices

### Model Configuration
- **Architecture**: Llama-3-8b-chat-hf (8B parameters)
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Key-Value Heads**: 8
- **Layers**: 32
- **Vocabulary Size**: 128,256
- **Max Sequence Length**: 8192

## Testing

### Verification Script
```bash
# Run the verification script
./test_executorch_setup.sh
```

### Sample Prompts
Test the integration with these sample prompts:
- "What is baseball?"
- "Hello, how are you?"
- "Explain artificial intelligence"
- "What is machine learning?"

## Troubleshooting

### Common Issues

1. **Context Binary Version Mismatch**
   - Ensure you're using v79 context binaries
   - Verify SoC Model-69 compatibility

2. **Memory Issues**
   - Ensure device has 16GB RAM
   - Close other memory-intensive apps

3. **Native Library Loading**
   - Check if native library is properly built
   - Verify NDK version compatibility

4. **Tokenizer Issues**
   - Ensure tokenizer.model is properly downloaded
   - Verify file placement in assets directory

### Debug Information

Enable debug logging to see detailed information:
```kotlin
// The app will log detailed information about:
// - Model initialization
// - Context binary verification
// - Token generation process
// - ExecuTorch runtime status
```

## Performance Considerations

- **Memory Usage**: 16GB RAM recommended for optimal performance
- **Processing Time**: First inference may take longer due to model loading
- **Temperature**: Lower values (0.1-0.5) for more focused responses
- **Max Tokens**: Adjust based on desired response length

## References

- [ExecuTorch GitHub](https://github.com/pytorch/executorch)
- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/)
- [Qualcomm Backend Guide](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)
- [Qualcomm AI HUB](https://aihub.qualcomm.com/)
- [Llama-v3-8B-Chat Model](https://huggingface.co/qualcomm/Llama-v3-8B-Chat)
- [ExecuTorch Qualcomm Examples](https://github.com/pytorch/executorch/tree/main/examples/qualcomm)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For support and questions:
- Create an issue in this repository
- Check the ExecuTorch documentation
- Visit the Qualcomm AI HUB community forums
