# Method: Tapered Architectures

Tapered (or "funnel") architectures are neural network designs where the number of neurons decreases in subsequent layers.

## Implementation
Instead of a constant width (e.g., `[40, 40, 40, 40]`), a tapered network follows a descending pattern, such as:
`[120, 100, 80, 60, 40, 20]`

## Rationale
- **Feature Compression**: Helps in condensing high-dimensional physical features into essential components.
- **Improved Accuracy**: Experiments in [[Heat2D_Analysis]] showed that funnel architectures provide a better accuracy-to-parameter ratio for Laplacian problems.

## References
- Standard practice in [[Note_01_Framework]] and [[Heat2D_Analysis]].
