// In app/src/main/java/com/example/edgeai/ui/MainViewModel.kt
package com.example.edgeai.ui.theme

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.edgeai.ml.CLIPInference
import kotlinx.coroutines.launch

// Represents the current state of our UI
data class VqaUiState(
    val selectedImage: Bitmap? = null,
    val question: String = "What color is the car?",
    val answer: String = "",
    val isLoading: Boolean = false
)

class MainViewModel : ViewModel() {
    var uiState by mutableStateOf(VqaUiState())
        private set

    private lateinit var CLIPInference: CLIPInference

    fun initialize(context: Context) {
        // We only initialize the executor once
        if (!::CLIPInference.isInitialized) {
            CLIPInference = CLIPInference(context)
        }
    }

    fun onImageSelected(bitmap: Bitmap) {
        uiState = uiState.copy(selectedImage = bitmap, answer = "")
    }

    fun onQuestionChanged(newQuestion: String) {
        uiState = uiState.copy(question = newQuestion)
    }

    fun askQuestion() {
        val image = uiState.selectedImage ?: return
        val question = uiState.question

        // Launch a coroutine to run the model without freezing the UI
        viewModelScope.launch {
            uiState = uiState.copy(isLoading = true)
            val results = CLIPInference.runInference(image)
            val result = if (results != null) {
                "Inference completed. Found ${results.size} output tensors."
            } else {
                "Inference failed or returned no results."
            }
            uiState = uiState.copy(answer = result, isLoading = false)
        }
    }
}
    