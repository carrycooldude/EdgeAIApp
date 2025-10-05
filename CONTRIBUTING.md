# Contributing to EdgeAI

Thank you for your interest in contributing to EdgeAI! This document provides guidelines for contributing to our project.

## 🚀 Getting Started

### Prerequisites

- Android Studio (Arctic Fox or later)
- Android NDK (25.1.8937393)
- Qualcomm QNN SDK
- Git

### Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/EdgeAIApp.git
   cd EdgeAIApp
   ```

2. **Set Up Development Environment**
   ```bash
   # Install QNN libraries (see app/src/main/jniLibs/README.md)
   # Build the project
   ./gradlew assembleDebug
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Code Style Guidelines

### Kotlin Code Style

- Follow [official Kotlin coding conventions](https://kotlinlang.org/docs/coding-conventions.html)
- Use meaningful variable and function names
- Add KDoc comments for public APIs
- Use `camelCase` for variables and functions
- Use `PascalCase` for classes

```kotlin
/**
 * Runs LLaMA inference on the given input text
 * @param inputText The text to process
 * @param maxTokens Maximum number of tokens to generate
 * @return Generated response text
 */
fun runInference(inputText: String, maxTokens: Int = 100): String? {
    // Implementation
}
```

### C++ Code Style

- Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use meaningful variable names
- Add comprehensive comments
- Use `snake_case` for variables and functions
- Use `PascalCase` for classes

```cpp
/**
 * Initializes the QNN inference engine
 * @return true if initialization successful, false otherwise
 */
bool initialize_qnn_inference() {
    // Implementation
}
```

### Java Code Style

- Follow [Android coding standards](https://source.android.com/setup/contribute/code-style)
- Use meaningful names
- Add JavaDoc comments
- Use `camelCase` for variables and functions

## 🧪 Testing Guidelines

### Unit Tests

- Write unit tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names

```kotlin
@Test
fun `runInference should return valid response for valid input`() {
    // Given
    val inputText = "Hello world"
    val maxTokens = 50
    
    // When
    val result = llamaInference.runInference(inputText, maxTokens)
    
    // Then
    assertThat(result).isNotNull()
    assertThat(result).isNotEmpty()
}
```

### Integration Tests

- Test JNI integration
- Test QNN library integration
- Test end-to-end inference pipeline

### Performance Tests

- Benchmark inference times
- Test memory usage
- Validate NPU acceleration

## 📋 Pull Request Process

### Before Submitting

1. **Run Tests**
   ```bash
   ./gradlew test
   ./gradlew connectedAndroidTest
   ```

2. **Check Code Style**
   ```bash
   ./gradlew ktlintCheck
   ./gradlew detekt
   ```

3. **Build Successfully**
   ```bash
   ./gradlew assembleDebug
   ```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Performance Impact
- [ ] No performance impact
- [ ] Performance improvement
- [ ] Performance regression (explain)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass
```

## 🐛 Bug Reports

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Device: [e.g., Samsung Galaxy S23]
- Android Version: [e.g., Android 13]
- App Version: [e.g., 1.0.0]

## Logs
```
Paste relevant logs here
```

## Screenshots
If applicable, add screenshots
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Other approaches you've considered

## Additional Context
Any other relevant information
```

## 🏷️ Issue Labels

We use the following labels for issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `priority: high`: High priority
- `priority: medium`: Medium priority
- `priority: low`: Low priority

## 📚 Documentation

### Code Documentation

- Add KDoc/JavaDoc comments for public APIs
- Update README.md for new features
- Add inline comments for complex logic
- Document configuration changes

### API Documentation

- Document all public methods
- Include parameter descriptions
- Include return value descriptions
- Add usage examples

## 🔒 Security

### Security Guidelines

- Never commit sensitive information (API keys, passwords)
- Use secure coding practices
- Validate all inputs
- Handle errors gracefully
- Follow Android security best practices

### Reporting Security Issues

If you discover a security vulnerability, please:

1. **DO NOT** create a public issue
2. Email security concerns to [your-email]
3. Include detailed information about the vulnerability
4. Allow time for the issue to be addressed before public disclosure

## 🎯 Development Workflow

### Branch Naming

- `feature/feature-name`: New features
- `bugfix/bug-description`: Bug fixes
- `hotfix/critical-fix`: Critical fixes
- `docs/documentation-update`: Documentation updates
- `refactor/refactoring-description`: Code refactoring

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(llama): add context-aware response generation
fix(qnn): resolve memory leak in native inference
docs(readme): update installation instructions
```

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Help others learn and grow

### Getting Help

- Check existing issues and discussions
- Ask questions in GitHub Discussions
- Join our community chat (if available)
- Read the documentation

## 📞 Contact

- **Maintainer**: [Your Name]
- **Email**: [Your Email]
- **GitHub**: [Your GitHub Profile]

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to EdgeAI! 🚀
