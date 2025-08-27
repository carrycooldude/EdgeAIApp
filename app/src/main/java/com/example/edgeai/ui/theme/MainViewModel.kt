// In app/src/main/java/com/example/edgeai/ui/MainViewModel.kt
package com.example.edgeai.ui.theme

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.edgeai.ml.VqaExecutor
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

    private lateinit var vqaExecutor: VqaExecutor

    fun initialize(context: Context) {
        // We only initialize the executor once
        if (!::vqaExecutor.isInitialized) {
            vqaExecutor = VqaExecutor(context)
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
            val result = vqaExecutor.execute(image, question)
            uiState = uiState.copy(answer = result, isLoading = false)
        }
    }
}
    