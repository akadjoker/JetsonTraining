# Contributing to Autonomous Driving System

First off, thank you for considering contributing to our Autonomous Driving System project! It's people like you that make this project such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Issue Tracking](#issue-tracking)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [your.email@example.com](mailto:your.email@example.com).

## Getting Started

### Development Environment Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/akadjoker/JetsonTraining.git
   cd JetsonTraining
   ```
3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/originalowner/JetsonTraining.git
   ```
4. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

## Development Workflow

We follow a simplified GitFlow workflow:

1. `main`: Production-ready code
2. `develop`: Integration branch for features
3. Feature branches: For new features and non-emergency bug fixes
4. Hotfix branches: For critical bugs in production

### Branch Naming Convention

- Feature branches: `feature/short-description`
- Bug fix branches: `bugfix/short-description`
- Hotfix branches: `hotfix/short-description`
- Release branches: `release/vX.Y.Z`

### Workflow Steps

1. Sync your fork with the upstream repository:
   ```bash
   git checkout develop
   git pull upstream develop
   ```

2. Create a new branch for your work:
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. Make your changes, committing regularly with good commit messages

4. Push your branch to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```

5. Create a Pull Request from your fork to the upstream repository

## Commit Message Guidelines

We follow a modified version of the [Conventional Commits](https://www.conventionalcommits.org/) standard:

```
<type>(<scope>): <short summary>

<body>

<footer>
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Scope

The scope should be the name of the module affected (e.g., `data`, `model`, `training`, `visualization`).

### Examples

```
feat(model): add batch normalization to convolutional layers

Add batch normalization after each convolutional layer to improve 
training stability and convergence speed.

Resolves: #123
```

```
fix(training): correct data augmentation transformation

The horizontal flip was not correctly inverting the steering angle,
which caused the model to learn incorrect behavior on turns.

Fixes: #456
```

## Pull Request Process

1. Update the README.md or relevant documentation with details of changes
2. Update the requirements.txt if you've added dependencies
3. Make sure all tests pass
4. Ensure your code follows our coding standards
5. Include relevant unit/integration tests for your changes
6. The PR must receive approval from at least one maintainer
7. Maintainers will merge the PR when it's ready

## Coding Standards

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style with a few modifications:

- Line length: 100 characters maximum
- Use 4 spaces for indentation
- Use docstrings in the NumPy/SciPy format

### Example Docstring Format

```python
def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.
    
    This function does something with the input parameters.
    
    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str
        The second parameter.
    
    Returns
    -------
    bool
        True if successful, False otherwise.
        
    Examples
    --------
    >>> function_with_types_in_docstring(1, '2')
    True
    """
    return True
```

## Testing Guidelines

- Write tests for all new features and bug fixes
- Aim for at least 80% code coverage
- Tests should be written using pytest
- Test files should be named `test_*.py`
- Run the full test suite before submitting a PR

```bash
pytest -v
```

## Documentation Guidelines

- Keep the API documentation up-to-date with code changes
- Document complex algorithms and design decisions
- Add examples for new features
- Use diagrams when appropriate to explain complex systems
- Update the model architecture documentation when changes are made

## Issue Tracking

### Issue Templates

We provide templates for:
- Bug reports
- Feature requests
- Model improvement proposals

### Issue Labeling

Use the following labels:
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation related issues
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `model`: Related to neural network architecture
- `data`: Related to dataset or preprocessing
- `training`: Related to training process
- `testing`: Related to testing/evaluation
- `visualization`: Related to visual outputs

### Issue Workflow

1. **New**: Issue has been created
2. **Triage**: Issue is being evaluated for validity and priority
3. **Accepted**: Issue has been accepted for resolution
4. **In Progress**: Someone is working on it
5. **Review**: Work is complete and under review
6. **Done**: Issue has been resolved

Thank you for contributing to our project!
