---
sidebar_position: 1
---

# Qualcomm AI HUB Setup Guide

This guide will walk you through setting up Qualcomm AI HUB for compiling Llama3.2-1B models into context binaries.

## Prerequisites

- Qualcomm AI HUB account
- QNN SDK v2.37.0 or later
- Python 3.8+
- Git

## Step 1: Account Setup

1. **Create Account**: Visit [Qualcomm AI HUB](https://aihub.qualcomm.com/)
2. **Verify Email**: Complete email verification
3. **Access Dashboard**: Log into your account

## Step 2: Download QNN SDK

1. **Navigate to Downloads**: Go to SDK downloads section
2. **Select Version**: Choose QNN SDK v2.37.0 or later
3. **Download**: Get the appropriate package for your OS

```bash
# Extract the SDK
tar -xzf qnn-sdk-v2.37.0.tar.gz
cd qnn-sdk-v2.37.0
```

## Step 3: Environment Setup

### Windows (PowerShell)
```powershell
# Set environment variables
$env:QNN_SDK_ROOT = "C:\path\to\qnn-sdk-v2.37.0"
$env:PATH += ";$env:QNN_SDK_ROOT\bin"

# Verify installation
qnn-onnx-converter --version
```

### Linux/macOS
```bash
# Add to ~/.bashrc or ~/.zshrc
export QNN_SDK_ROOT="/path/to/qnn-sdk-v2.37.0"
export PATH="$QNN_SDK_ROOT/bin:$PATH"

# Reload shell
source ~/.bashrc
```

## Step 4: Model Compilation

### Prepare Model
1. **Download Llama3.2-1B**: Get the model files
2. **Convert Format**: Convert to ONNX if needed
3. **Upload to AI HUB**: Use the web interface

### Compile Context Binaries
```bash
# Use QNN tools to compile
qnn-onnx-converter \
  --input_network model.onnx \
  --output_path ./context_binaries \
  --target_arch aarch64-android \
  --hexagon_version v79
```

## Step 5: Integration

### Android Project Setup
1. **Copy Libraries**: Add QNN libraries to `app/src/main/jniLibs/`
2. **Update CMakeLists.txt**: Link QNN libraries
3. **Configure Gradle**: Add native library dependencies

### Verification
```bash
# Test context binary loading
adb shell "cd /data/local/tmp && ./test_qnn_context"
```

## Troubleshooting

### Common Issues

**Issue**: Context binary loading fails
**Solution**: Verify hexagon version matches device (v79 for newer devices)

**Issue**: Model compilation errors
**Solution**: Check ONNX model compatibility and QNN SDK version

**Issue**: Performance issues
**Solution**: Ensure proper ARM64-v8a architecture targeting

## Next Steps

- [Project Structure](../technical/project-structure)
- [Release Notes](../releases/release-notes-v1-4-0)
