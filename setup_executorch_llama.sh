#!/bin/bash

# ExecuTorch Llama-3-8b-chat-hf Setup Script
# This script helps set up ExecuTorch with Qualcomm HTP backend for Llama-3-8b-chat-hf
# Based on official ExecuTorch Qualcomm examples

echo "🚀 Setting up ExecuTorch Llama-3-8b-chat-hf with Qualcomm HTP backend"
echo "=================================================================="

# Check if we're in the right directory
if [ ! -f "app/build.gradle.kts" ]; then
    echo "❌ Error: Please run this script from the EdgeAI project root directory"
    exit 1
fi

echo "✅ Found EdgeAI project directory"

# Step 1: Install ExecuTorch dependencies
echo ""
echo "📦 Step 1: Installing ExecuTorch dependencies..."
echo "Installing ExecuTorch Python package for model export..."

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "✅ Python3 found, installing ExecuTorch..."
    pip3 install executorch
    pip3 install torch
    pip3 install transformers
    pip3 install optimum-executorch
else
    echo "⚠️ Python3 not found. Please install Python3 and ExecuTorch manually:"
    echo "   pip3 install executorch torch transformers optimum-executorch"
fi

# Step 2: Create directory structure for models
echo ""
echo "📁 Step 2: Creating directory structure for Llama-3-8b-chat-hf model..."

MODEL_DIR="app/src/main/assets/models/Llama-3-8b-chat-hf"
TOKENIZER_DIR="app/src/main/assets/tokenizer"
CONTEXT_BINARIES_DIR="app/src/main/assets/context_binaries"

mkdir -p "$MODEL_DIR"
mkdir -p "$TOKENIZER_DIR"
mkdir -p "$CONTEXT_BINARIES_DIR"

echo "✅ Created directories:"
echo "   - $MODEL_DIR"
echo "   - $TOKENIZER_DIR"
echo "   - $CONTEXT_BINARIES_DIR"

# Step 3: Create setup instructions
echo ""
echo "📋 Step 3: Setup Instructions for Llama-3-8b-chat-hf"
echo "=================================================="
echo ""
echo "To complete the setup, you need to:"
echo ""
echo "1. 🔐 Create Qualcomm AI HUB Account:"
echo "   - Visit: https://aihub.qualcomm.com/"
echo "   - Create an account and get access to AI HUB"
echo ""
echo "2. 📥 Export Llama-3-8b-chat-hf Context Binaries:"
echo "   - Follow instructions at: https://huggingface.co/qualcomm/Llama-v3-8B-Chat"
echo "   - Export context binaries using Qualcomm AI HUB"
echo "   - This will take some time to complete"
echo "   - Place the exported binaries in: $CONTEXT_BINARIES_DIR"
echo ""
echo "3. 🔤 Download Llama 3 Tokenizer:"
echo "   - Visit: https://github.com/meta-llama/llama-models/blob/main/README.md"
echo "   - Download tokenizer.model file"
echo "   - Place it in: $TOKENIZER_DIR"
echo ""
echo "4. ✅ Verify Context Binary Version:"
echo "   - Ensure context binaries are version v79"
echo "   - Compatible with SoC Model-69"
echo "   - Required files: libQnnHtp.so, libQnnHtpV79Stub.so, libQnnSystem.so"
echo ""

# Step 4: Create example usage code
echo ""
echo "💻 Step 4: Example Usage Code"
echo "============================"
echo ""
echo "Here's how to use the ExecuTorch Llama-3-8b-chat-hf integration:"
echo ""
echo "```kotlin"
echo "// Initialize ExecuTorch Llama-3-8b-chat-hf"
echo "val llamaInference = LLaMAInference(context)"
echo ""
echo "// Initialize with model paths"
echo "val success = llamaInference.initializeExecuTorchLlama("
echo "    modelPath = \"path/to/model\","
echo "    tokenizerPath = \"path/to/tokenizer.model\","
echo "    contextBinariesPath = \"path/to/context_binaries\""
echo ")"
echo ""
echo "if (success) {"
echo "    // Generate response"
echo "    val response = llamaInference.generateExecuTorchLlama("
echo "        prompt = \"What is baseball?\","
echo "        maxTokens = 128,"
echo "        temperature = 0.8f"
echo "    )"
echo "    println(\"Response: \$response\")"
echo "}"
echo "```"
echo ""

# Step 5: Create build script for testing
echo ""
echo "🔨 Step 5: Build and Test Script"
echo "================================"
echo ""
echo "To build and test the app:"
echo ""
echo "1. Build the project:"
echo "   ./gradlew assembleDebug"
echo ""
echo "2. Install on device:"
echo "   ./gradlew installDebug"
echo ""
echo "3. Run the app and test with prompt: \"What is baseball?\""
echo ""

# Step 6: Create verification script
echo ""
echo "🔍 Step 6: Verification Checklist"
echo "=================================="
echo ""
echo "Before running the app, verify:"
echo ""
echo "✅ ExecuTorch dependencies installed"
echo "✅ Qualcomm AI HUB account created"
echo "✅ Context binaries exported (v79, SoC Model-69)"
echo "✅ Tokenizer.model downloaded"
echo "✅ All files placed in correct directories"
echo "✅ Device has Qualcomm SoC with AI Engine Direct support"
echo "✅ Device has sufficient RAM (16GB recommended)"
echo ""

# Create a simple test script
cat > test_executorch_setup.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing ExecuTorch Llama-3-8b-chat-hf Setup"
echo "=============================================="

# Check if required directories exist
REQUIRED_DIRS=(
    "app/src/main/assets/models/Llama-3-8b-chat-hf"
    "app/src/main/assets/tokenizer"
    "app/src/main/assets/context_binaries"
)

echo "Checking required directories..."
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir exists"
    else
        echo "❌ $dir missing"
    fi
done

# Check for required files
echo ""
echo "Checking for required files..."

# Check for tokenizer
if [ -f "app/src/main/assets/tokenizer/tokenizer.model" ]; then
    echo "✅ tokenizer.model found"
else
    echo "❌ tokenizer.model missing"
fi

# Check for context binaries
CONTEXT_FILES=(
    "libQnnHtp.so"
    "libQnnHtpV79Stub.so"
    "libQnnSystem.so"
)

for file in "${CONTEXT_FILES[@]}"; do
    if [ -f "app/src/main/assets/context_binaries/$file" ]; then
        echo "✅ $file found"
    else
        echo "❌ $file missing"
    fi
done

echo ""
echo "Setup verification complete!"
EOF

chmod +x test_executorch_setup.sh

echo "✅ Created test script: test_executorch_setup.sh"
echo ""
echo "🎉 Setup script completed!"
echo ""
echo "Next steps:"
echo "1. Run: ./test_executorch_setup.sh (to verify setup)"
echo "2. Follow the setup instructions above"
echo "3. Build and test the app"
echo ""
echo "For more information, visit:"
echo "- ExecuTorch: https://github.com/pytorch/executorch"
echo "- Qualcomm AI HUB: https://aihub.qualcomm.com/"
echo "- Llama-3-8b-chat-hf: https://huggingface.co/qualcomm/Llama-v3-8B-Chat"
