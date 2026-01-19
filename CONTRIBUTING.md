# Contributing to ChatBot

Thank you for considering contributing to ChatBot! We welcome all kinds of contributions: bug reports, improvements, new features, documentation, etc.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Style Guidelines](#style-guidelines)
6. [Commit Messages](#commit-messages)
7. [Pull Requests](#pull-requests)
8. [Reporting Bugs](#reporting-bugs)
9. [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to abide by its terms.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone git@github.com:YOUR_USERNAME/ChatBot.git
   cd ChatBot
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/ob-labs/ChatBot.git
   ```
4. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## How to Contribute

### Reporting Bugs

- Ensure you're using the latest version
- Check if the issue has already been reported
- Include your environment details (OS, Python version, etc.)
- Provide clear steps to reproduce the issue
- Include error messages or logs if applicable

### Suggesting Enhancements

- Search existing issues before filing a new one
- Use clear titles and descriptions
- Explain the motivation behind your suggestion
- Describe the expected behavior

### Code Contributions

- Write clear, maintainable code
- Add tests for new features or bug fixes
- Update documentation as needed
- Ensure all tests pass before submitting

## Development Setup

1. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

2. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Run the application:
   ```bash
   poetry run streamlit run src/frontend/chat_ui.py
   ```

## Style Guidelines

- Follow PEP 8 for Python code style
- Use meaningful variable and function names
- Add docstrings to public functions and classes
- Keep functions focused and small
- Write comments for complex logic

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 50 characters
- Reference issues and pull requests when applicable
- Provide detailed body if necessary

Example:
```
Add vector search functionality

- Implement vector similarity search in OceanBase
- Add configuration for embedding dimensions
- Update documentation with usage examples

Fixes #123
```

## Pull Requests

1. Make sure your branch is up to date with the main branch:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Push your commits to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Open a pull request on GitHub:
   - Use a clear title and description
   - Reference related issues
   - Include screenshots or examples if applicable
   - Be responsive to review comments

4. Make requested changes:
   - Address review feedback promptly
   - Keep commits focused and logical
   - Update your PR description if needed

## Reporting Bugs

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: OS, Python version, package versions
- **Logs**: Relevant error messages or logs

## Suggesting Enhancements

When suggesting enhancements:

- **Description**: Clear description of the enhancement
- **Use Case**: Explain why this enhancement would be useful
- **Proposed Solution**: If you have ideas on how to implement it
- **Alternatives**: Other solutions you've considered

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Thank You

Thank you for contributing to ChatBot! Your contributions make this project better for everyone.
