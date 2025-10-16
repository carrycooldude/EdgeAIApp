---
sidebar_position: 1
---

# Project Structure

The EdgeAI project follows a well-organized structure that separates concerns and makes the codebase maintainable.

## Directory Overview

```
EdgeAI/
├── app/                          # Android application
│   ├── src/main/
│   │   ├── java/                 # Kotlin/Java source code
│   │   ├── cpp/                  # Native C++ code
│   │   └── assets/               # App assets
│   └── build.gradle.kts          # Build configuration
├── docs/                         # EdgeAI Documentation Platform
│   ├── docs/                     # Documentation content
│   ├── src/                      # EdgeAI source files
│   ├── static/                   # Static assets
│   └── docusaurus.config.ts     # EdgeAI configuration
├── backup_qnn_libs/              # QNN library backups
└── build/                        # Build outputs
```

## Key Components

### Android App (`app/`)
- **Kotlin/Java**: Main application logic and UI
- **C++**: Native inference engine with ExecuTorch + QNN
- **Assets**: Model files and resources

### Documentation (`docs/`)
- **EdgeAI Documentation Platform**: Modern documentation framework
- **Organized content**: Technical docs, setup guides, releases
- **Responsive design**: Mobile-friendly documentation

### Native Libraries (`backup_qnn_libs/`)
- **ARM64-v8a**: Optimized for modern Android devices
- **QNN libraries**: Qualcomm Neural Processing SDK
- **Version-specific**: v79 context binaries for SoC Model-69

## Architecture Benefits

1. **Separation of Concerns**: Clear boundaries between app, docs, and libraries
2. **Maintainability**: Easy to locate and modify specific components
3. **Scalability**: Structure supports future expansion
4. **Documentation**: Self-documenting through organized docs structure

## Development Workflow

1. **App Development**: Work in `app/` directory
2. **Documentation**: Update files in `docs/docs/` directory
3. **Native Code**: Modify C++ files in `app/src/main/cpp/`
4. **Build & Test**: Use Gradle build system
5. **Deploy**: Automatic documentation deployment via GitHub Actions
