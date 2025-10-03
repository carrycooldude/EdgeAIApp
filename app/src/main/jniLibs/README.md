# Qualcomm QNN Native Libraries

This directory should contain the Qualcomm QNN native libraries for Android.

## Required Libraries

### For ARM64-v8a (64-bit devices):
- `libQnnHtp.so` - Hexagon Tensor Processor library
- `libQnnSystem.so` - QNN System library
- `libQnnHtpV69Stub.so` - QNN HTP V69 stub
- `libQnnHtpV73Stub.so` - QNN HTP V73 stub

### For ARMv7 (32-bit devices):
- `libQnnHtp.so` - Hexagon Tensor Processor library
- `libQnnSystem.so` - QNN System library
- `libQnnHtpV69Stub.so` - QNN HTP V69 stub
- `libQnnHtpV73Stub.so` - QNN HTP V73 stub

## How to Get the Libraries

1. **Download Qualcomm Neural Processing SDK** from [Qualcomm Developer](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
2. **Extract the libraries** from the SDK package
3. **Copy the appropriate libraries** to the respective architecture folders
4. **Ensure proper permissions** are set

## Note

These libraries are **not included** in the Git repository due to their large size and licensing restrictions. You need to obtain them separately from Qualcomm.

## Alternative: Use Pre-built APK

If you don't have access to the QNN libraries, you can use the pre-built APK:
- `EdgeAI-Real-LLaMA-Inference.apk` (available in releases)

This APK contains all the necessary libraries and is ready to run on Qualcomm devices.
