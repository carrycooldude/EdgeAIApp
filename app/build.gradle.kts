plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    id("maven-publish")
    id("signing")
}

android {
    namespace = "com.example.edgeai"
    compileSdk = 36
    ndkVersion = "25.1.8937393"

    defaultConfig {
        applicationId = "com.example.edgeai"
        minSdk = 24
        targetSdk = 36
        versionCode = 4
        versionName = "1.4.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        // Samsung S25 Ultra compatibility
        ndk {
            abiFilters += listOf("arm64-v8a") // aarch64-android for v79 context binaries
        }
        
        // Samsung S25 Ultra page size compatibility
        packaging {
            jniLibs {
                useLegacyPackaging = false
                // Add Samsung S25 Ultra specific packaging options
                pickFirsts += listOf("**/libedgeai_qnn.so")
            }
        }
        
        // Samsung S25 Ultra specific configurations
        manifestPlaceholders["samsung_s25_ultra_compat"] = "true"
        
        // Memory management for Samsung S25 Ultra
        multiDexEnabled = true
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // Optimize for release
            packaging {
                jniLibs {
                    useLegacyPackaging = false
                }
                resources {
                    excludes += listOf(
                        "**/consolidated.00.pth",
                        "**/consolidated.*.pth",
                        "**/*.bin",
                        "**/*.safetensors"
                    )
                }
            }
        }
        debug {
            isMinifyEnabled = false
            // Enable 16 KB page size compatibility for debug builds
            packaging {
                jniLibs {
                    useLegacyPackaging = false
                }
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    ndkVersion = "25.1.8937393"
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.1")
    
    // Additional ML dependencies for LLaMA
    implementation("com.google.code.gson:gson:2.10.1")
    implementation("org.apache.commons:commons-lang3:3.12.0")
}

// GitHub Package Publishing Configuration
publishing {
    publications {
        create<MavenPublication>("release") {
            groupId = "com.carrycooldude"
            artifactId = "edgeai-llama"
            version = "1.2.0"
            
            // Publish the APK as the main artifact
            artifact("${layout.buildDirectory.get()}/outputs/apk/release/app-release.apk") {
                classifier = "apk"
                extension = "apk"
            }
            
            // Publish source code
            artifact("${layout.buildDirectory.get()}/outputs/sources/release") {
                classifier = "sources"
                extension = "jar"
            }
            
            // Add POM metadata
            pom {
                name.set("EdgeAI LLaMA")
                description.set("EdgeAI LLaMA Model Integration with Qualcomm QNN NPU - Working AI responses on mobile devices")
                url.set("https://github.com/carrycooldude/EdgeAIApp-ExecuTorch")
                
                licenses {
                    license {
                        name.set("MIT License")
                        url.set("https://opensource.org/licenses/MIT")
                    }
                }
                
                developers {
                    developer {
                        id.set("carrycooldude")
                        name.set("CarryCoolDude")
                        email.set("carrycooldude@example.com")
                    }
                }
                
                scm {
                    connection.set("scm:git:git://github.com/carrycooldude/EdgeAIApp-ExecuTorch.git")
                    developerConnection.set("scm:git:ssh://github.com:carrycooldude/EdgeAIApp-ExecuTorch.git")
                    url.set("https://github.com/carrycooldude/EdgeAIApp-ExecuTorch")
                }
            }
        }
    }
    
    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/carrycooldude/EdgeAIApp-ExecuTorch")
            credentials {
                username = project.findProperty("gpr.user") as String? ?: System.getenv("GITHUB_ACTOR")
                password = project.findProperty("gpr.key") as String? ?: System.getenv("GITHUB_TOKEN")
            }
        }
    }
}

// Signing configuration for releases
signing {
    val signingKeyId = project.findProperty("signing.keyId") as String?
    val signingKey = project.findProperty("signing.key") as String?
    val signingPassword = project.findProperty("signing.password") as String?
    
    if (signingKeyId != null && signingKey != null && signingPassword != null) {
        useInMemoryPgpKeys(signingKeyId, signingKey, signingPassword)
        sign(publishing.publications["release"])
    }
}