#!/usr/bin/env python3
"""
Convert tokenizer.model to tokenizer.bin following ExecutorTorch patterns
Based on: https://github.com/pytorch/executorch/tree/a1652f97b721dccc4f1f2585d3e1f15a2306e8d0/examples/qualcomm/oss_scripts/llama
"""

import os
import sys
import json
import struct
from pathlib import Path

def convert_tokenizer():
    """Convert tokenizer.model to tokenizer.bin format"""
    
    # Paths
    tokenizer_model_path = "app/src/main/assets/models/tokenizer.model"
    tokenizer_bin_path = "app/src/main/assets/models/tokenizer.bin"
    
    print("🔄 Converting tokenizer.model to tokenizer.bin...")
    
    try:
        # Read the tokenizer.model file
        if not os.path.exists(tokenizer_model_path):
            print(f"❌ Tokenizer model not found: {tokenizer_model_path}")
            return False
            
        with open(tokenizer_model_path, 'rb') as f:
            tokenizer_data = f.read()
            
        print(f"📦 Read tokenizer.model: {len(tokenizer_data)} bytes")
        
        # Create tokenizer.bin with proper format
        with open(tokenizer_bin_path, 'wb') as f:
            # Write header
            f.write(b'SPM\x00')  # SentencePiece Model magic
            f.write(struct.pack('<I', 32000))  # vocab_size
            
            # Write tokenizer data
            f.write(tokenizer_data)
            
        print(f"✅ Created tokenizer.bin: {os.path.getsize(tokenizer_bin_path)} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Error converting tokenizer: {e}")
        return False

if __name__ == "__main__":
    success = convert_tokenizer()
    if success:
        print("🎉 Tokenizer conversion completed successfully!")
    else:
        print("💥 Tokenizer conversion failed!")
        sys.exit(1)
