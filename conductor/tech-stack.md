# Tech Stack

## Core Language & Frameworks
- **Python 3.x:** Primary programming language.
- **PyTorch (torch):** Main framework for deep learning and automatic differentiation.
- **NumPy:** Numerical computing and data manipulation.

## Data Analysis & Processing
- **Pandas:** Data manipulation and analysis for experiment results.

## Visualization & Imaging
- **Matplotlib:** Core plotting library for results and analysis.
- **Seaborn:** Statistical data visualization based on matplotlib.
- **Pillow:** Image processing and GIF generation for transient physical simulations.

## Utilities
- **tqdm:** Progress bars for monitoring training loops.
- **Data Management**: `CSV` for experiment logging.
- **Logging Schema**: Includes `n_points` to track dataset size in `results.csv`.
- **JSON/Markdown:** For configuration state and documentation.

## Development Environment
- **Virtual Environment:** Recommended (`.venv` or `tesi`).
- **Precision:** `torch.float64` as default for numerical stability in physics-informed tasks.

---

### Deviations & Updates

#### 2026-01-22
- Added **Pandas** and **Seaborn** to support the new `Heat2D/analyze_results.py` script for advanced data analysis and visualization of experiment results.