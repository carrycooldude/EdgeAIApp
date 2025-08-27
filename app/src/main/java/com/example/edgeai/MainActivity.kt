package com.example.edgeai

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.preference.isNotEmpty
import com.example.edgeai.ui.MainViewModel
import com.example.edgeai.ui.theme.EdgeAITheme

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        viewModel.initialize(applicationContext) // Initialize the model executor

        setContent {
            EdgeAITheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MainScreen(viewModel)
                }
            }
        }
    }
}

@Composable
fun MainScreen(viewModel: MainViewModel) {
    val uiState = viewModel.uiState
    val context = LocalContext.current

    // This launcher handles opening the phone's gallery to pick an image
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val bitmap = if (Build.VERSION.SDK_INT < 28) {
                MediaStore.Images.Media.getBitmap(context.contentResolver, it)
            } else {
                val source = ImageDecoder.createSource(context.contentResolver, it)
                ImageDecoder.decodeBitmap(source)
            }
            // Ensure bitmap is in a format the model can use
            val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            viewModel.onImageSelected(mutableBitmap)
        }
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = { TopAppBar(title = { Text("EdgeAI Multi-Modal Demo") }) }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .padding(innerPadding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Image Display and Picker Button
            if (uiState.selectedImage != null) {
                Image(
                    bitmap = uiState.selectedImage.asImageBitmap(),
                    contentDescription = "Selected Image",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(250.dp),
                    contentScale = ContentScale.Crop
                )
            }
            Button(onClick = { imagePickerLauncher.launch("image/*") }) {
                Text(if (uiState.selectedImage == null) "Select Image" else "Change Image")
            }

            // Question Input Field
            OutlinedTextField(
                value = uiState.question,
                onValueChange = { viewModel.onQuestionChanged(it) },
                label = { Text("Ask a question") },
                modifier = Modifier.fillMaxWidth()
            )

            // Action Button to Get Answer
            Button(
                onClick = { viewModel.askQuestion() },
                enabled = uiState.selectedImage != null && !uiState.isLoading
            ) {
                Text("Get Answer")
            }

            // Result Display Area
            if (uiState.isLoading) {
                CircularProgressIndicator()
            } else if (uiState.answer.isNotEmpty()) {
                Text(
                    text = "Answer: ${uiState.answer}",
                    style = MaterialTheme.typography.headlineSmall
                )
            }
        }
    }
}