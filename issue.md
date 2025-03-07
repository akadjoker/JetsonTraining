# GitHub Issue Templates

Place these files in the `.github/ISSUE_TEMPLATE/` directory of your repository.

## Bug Report Template

File: `bug_report.md`

```markdown
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## Steps To Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
What actually happened instead.

## Screenshots/Logs
If applicable, add screenshots or logs to help explain your problem.

## Environment
 - OS: [e.g. Ubuntu 18.00]
 - Hardware: [e.g. Jetson Nano 4GB]
 - Python version: [e.g. 3.8]
 - TensorFlow version: [e.g. 2.5]
 - Other relevant dependencies:

## Additional Context
Add any other context about the problem here.
```

## Feature Request Template

File: `feature_request.md`

```markdown
---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

## Problem Statement
Is your feature request related to a problem? Please describe.
A clear and concise description of what the problem is. E.g., I'm always frustrated when [...]

## Proposed Solution
A clear and concise description of what you want to happen.

## Alternative Solutions
A clear and concise description of any alternative solutions or features you've considered.

## Technical Implementation Details
If you have ideas about how this could be implemented, please share here.

## Benefits
What benefits would this feature bring to the project or users?

## Additional Context
Add any other context or screenshots about the feature request here.
```

## Model Improvement Template

File: `model_improvement.md`

```markdown
---
name: Model improvement
about: Suggest an improvement to the neural network model
title: '[MODEL] '
labels: model, enhancement
assignees: ''
---

## Aspect to Improve
Which aspect of the model needs improvement? 
[e.g., Accuracy on sharp turns, Inference speed, Generalization to new environments]

## Proposed Approach
Describe your suggested approach to improve the model.
[e.g., Different architecture, Additional data augmentation technique, New loss function]

## Expected Results
What results do you expect from this improvement?
[e.g., 15% reduction in MSE, Better performance in low-light conditions]

## Suggested Experiments
What experiments would you recommend to validate this improvement?
[e.g., Compare performance with and without the change, Test on specific datasets]

## Relevant Datasets
Are there specific datasets or driving sessions that highlight the current limitations?

## Implementation Considerations
Any considerations for implementing this change?
[e.g., Computational requirements, Compatibility issues]

## References
Any papers, articles, or other references that support this approach?
```

## Documentation Improvement Template

File: `documentation_improvement.md`

```markdown
---
name: Documentation improvement
about: Suggest an improvement to project documentation
title: '[DOCS] '
labels: documentation
assignees: ''
---

## Documentation Area
Which documentation needs improvement?
[e.g., README, API docs, Code comments, Tutorials]

## Current Issue
What is unclear or missing in the current documentation?

## Suggested Improvement
How could the documentation be improved?

## Why Is This Important?
Who would benefit from this improvement and how?

## Additional Context
Add any other context about the documentation issue here.
```

## Dataset Issue Template

File: `dataset_issue.md`

```markdown
---
name: Dataset issue
about: Report an issue with the training or test datasets
title: '[DATA] '
labels: data
assignees: ''
---

## Dataset Information
- Dataset version: [e.g., v1.0.0]
- Session ID(s): [e.g., session_20250227_203034]
- Number of affected samples: [approximate count]

## Issue Description
A clear description of the issue with the dataset.
[e.g., Missing data, Corrupted images, Incorrect labels]

## Impact on Training/Testing
How does this issue affect model training or testing?

## Proposed Resolution
If you have suggestions on how to resolve this issue, please share them.

## Screenshots/Examples
If applicable, add screenshots or examples of the problematic data.

## Additional Context
Add any other context about the dataset issue here.
```
