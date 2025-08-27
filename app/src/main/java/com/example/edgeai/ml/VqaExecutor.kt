package com.example.edgeai.ml

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class VqaExecutor(context: Context) {

    private val module: Module
    // This is a simplified tokenizer. A real implementation would parse the tokenizer.json file.
    private val tokenizer: Map<String, Long> = mapOf("<pad>" to 0, "what" to 1, "color" to 2, "is" to 3, "the" to 4, "car" to 5, "?" to 6)
    private val answers: List<String> = listOf("red", "blue", "green", "black", "white") // Example answers

    init {
        // This function loads the .pte model file from your app's assets
        module = LiteModuleLoader.load(assetFilePath(context, "vqa_model.pte"))
    }

    suspend fun execute(bitmap: Bitmap, question: String): String {
        // 1. Preprocess Image: Convert the user's image into a Tensor
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val imageTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        // 2. Preprocess Text: Convert the user's question into a Tensor
        val tokens = question.lowercase().split(" ").mapNotNull { tokenizer[it] }.toLongArray()
        val textTensor = Tensor.fromBlob(tokens, longArrayOf(1, tokens.size.toLong()))

        // 3. Run Inference: Feed the tensors into the model
        val output = module.forward(
            IValue.from(imageTensor),
            IValue.from(textTensor)
        ).toTensor()

        // 4. Post-process Output: Find the most likely answer
        val scores = output.dataAsFloatArray
        val maxIndex = scores.indices.maxByOrNull { scores[it] } ?: -1

        return if (maxIndex != -1) answers[maxIndex] else "Could not determine answer."
    }

    // Helper function to get the file path for a model in the assets folder
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
}
