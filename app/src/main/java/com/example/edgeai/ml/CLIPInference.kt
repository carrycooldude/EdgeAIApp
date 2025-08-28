package com.example.edgeai.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * CLIP Inference Engine for EdgeAI
 * Handles CLIP model loading and inference using QNN runtime
 */
class CLIPInference(private val context: Context) {

    companion object {
        private const val TAG = "CLIPInference"
        private const val MODEL_NAME = "openai_clip.dlc"

        // CLIP model input specifications
        private const val INPUT_WIDTH = 224
        private const val INPUT_HEIGHT = 224
        private const val INPUT_CHANNELS = 3

        // ImageNet normalization constants (used by CLIP)
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Load native QNN library
        init {
            try {
                System.loadLibrary("edgeai_qnn")
                Log.i(TAG, "✅ Native EdgeAI QNN library loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "❌ Failed to load native library: ${e.message}", e)
                throw RuntimeException("Failed to load EdgeAI QNN native library", e)
            }
        }
    }

    // Native method declarations (implemented in C++)
    private external fun nativeInitialize(modelPath: String): Boolean
    private external fun nativeRunInference(imageData: FloatArray, width: Int, height: Int): Map<String, FloatArray>?
    private external fun nativeGetInputShape(): IntArray
    private external fun nativeGetOutputInfo(): Array<String>
    private external fun nativeRelease()

    private var isInitialized = false
    private var modelFile: File? = null

    /**
     * Initialize the CLIP model for inference
     */
    fun initialize(): Boolean {
        try {
            Log.i(TAG, "🔧 Initializing CLIP inference engine...")

            // Check if model file exists in assets
            if (!checkModelExists()) {
                Log.e(TAG, "❌ Model file not found in assets/models/$MODEL_NAME")
                return false
            }

            // Copy model from assets to internal storage
            modelFile = copyModelFromAssets()
            Log.i(TAG, "📁 Model copied to: ${modelFile?.absolutePath}")

            // Initialize native QNN inference engine
            isInitialized = nativeInitialize(modelFile!!.absolutePath)

            if (isInitialized) {
                // Get model information
                try {
                    val inputShape = nativeGetInputShape()
                    val outputInfo = nativeGetOutputInfo()

                    Log.i(TAG, "📊 Model Info:")
                    Log.i(TAG, "   Input Shape: ${inputShape.joinToString(" x ")}")
                    Log.i(TAG, "   Output Count: ${outputInfo.size}")
                    outputInfo.forEachIndexed { index, info ->
                        Log.i(TAG, "   Output[$index]: $info")
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "⚠️ Could not retrieve model info: ${e.message}")
                }

                Log.i(TAG, "✅ CLIP model initialized successfully")
            } else {
                Log.e(TAG, "❌ Native CLIP initialization failed")
            }

            return isInitialized

        } catch (e: Exception) {
            Log.e(TAG, "❌ CLIP initialization error: ${e.message}", e)
            isInitialized = false
            return false
        }
    }

    /**
     * Run CLIP inference on input image
     */
    fun runInference(bitmap: Bitmap): Map<String, FloatArray>? {
        if (!isInitialized) {
            Log.e(TAG, "❌ CLIP model not initialized")
            return null
        }

        try {
            Log.i(TAG, "🔄 Preprocessing image for CLIP inference...")

            // Preprocess image
            val preprocessedData = preprocessImage(bitmap)
            Log.i(TAG, "✅ Image preprocessed: ${preprocessedData.size} float values")

            // Run native inference
            Log.i(TAG, "🚀 Executing QNN inference...")
            val results = nativeRunInference(preprocessedData, INPUT_WIDTH, INPUT_HEIGHT)

            if (results != null) {
                Log.i(TAG, "✅ Inference completed successfully")
                Log.i(TAG, "📊 Results summary:")
                results.forEach { (name, data) ->
                    Log.i(TAG, "   $name: ${data.size} values")
                }
            } else {
                Log.e(TAG, "❌ Inference returned null results")
            }

            return results

        } catch (e: Exception) {
            Log.e(TAG, "❌ Inference error: ${e.message}", e)
            return null
        }
    }

    /**
     * Check if model exists in assets
     */
    private fun checkModelExists(): Boolean {
        return try {
            val inputStream = context.assets.open("models/$MODEL_NAME")
            inputStream.close()
            true
        } catch (e: IOException) {
            false
        }
    }

    /**
     * Preprocess image for CLIP model input
     * - Resize to 224x224
     * - Normalize with ImageNet mean/std
     * - Convert to CHW format (Channel-Height-Width)
     */
    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        Log.i(TAG, "🖼️ Preprocessing image: ${bitmap.width}x${bitmap.height} -> ${INPUT_WIDTH}x${INPUT_HEIGHT}")

        // Resize image to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)

        // Extract pixel data
        val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
        resizedBitmap.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

        // Convert to normalized CHW format
        val inputData = FloatArray(INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH)

        for (i in pixels.indices) {
            val pixel = pixels[i]

            // Extract RGB values (0-255) and normalize to (0-1)
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            // Apply ImageNet normalization and store in CHW format
            inputData[i] = (r - MEAN[0]) / STD[0]                                    // R channel
            inputData[INPUT_HEIGHT * INPUT_WIDTH + i] = (g - MEAN[1]) / STD[1]       // G channel
            inputData[2 * INPUT_HEIGHT * INPUT_WIDTH + i] = (b - MEAN[2]) / STD[2]   // B channel
        }

        // Clean up
        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle()
        }

        Log.i(TAG, "✅ Image preprocessing completed")
        return inputData
    }

    /**
     * Copy DLC model from assets to internal storage
     */
    private fun copyModelFromAssets(): File {
        val modelFile = File(context.filesDir, MODEL_NAME)

        // If file already exists and is valid, use it
        if (modelFile.exists() && modelFile.length() > 0) {
            Log.i(TAG, "📁 Using existing model file: ${modelFile.absolutePath}")
            return modelFile
        }

        try {
            Log.i(TAG, "📥 Copying model from assets...")

            val inputStream = context.assets.open("models/$MODEL_NAME")
            val outputStream = FileOutputStream(modelFile)

            val buffer = ByteArray(8192) // 8KB buffer
            var totalBytes = 0
            var bytesRead: Int

            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
                totalBytes += bytesRead
            }

            inputStream.close()
            outputStream.close()

            Log.i(TAG, "✅ Model copied successfully: ${totalBytes} bytes")
            Log.i(TAG, "📁 Model location: ${modelFile.absolutePath}")

            return modelFile

        } catch (e: IOException) {
            Log.e(TAG, "❌ Failed to copy model from assets: ${e.message}", e)
            throw RuntimeException("Failed to copy CLIP model", e)
        }
    }

    /**
     * Release native resources and cleanup
     */
    fun release() {
        if (isInitialized) {
            try {
                Log.i(TAG, "🧹 Releasing CLIP inference resources...")
                nativeRelease()
                isInitialized = false
                Log.i(TAG, "✅ Resources released successfully")
            } catch (e: Exception) {
                Log.e(TAG, "❌ Error releasing resources: ${e.message}", e)
            }
        }

        // Clean up model file if needed
        modelFile = null
    }

    /**
     * Check if inference engine is ready
     */
    fun isReady(): Boolean = isInitialized

    /**
     * Get model input dimensions
     */
    fun getInputDimensions(): Triple<Int, Int, Int> = Triple(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)
}