# Multiple Regression script

**Version 1.0**
### Creator: Juhani Merilehto - @juhanimerilehto - Jyväskylä University of Applied Sciences (JAMK), Likes institute

![JAMK Likes Logo](./assets/likes_str_logo.png)

## Overview

Multiple Regression script for quantitative analysis. This Python-based tool enables comprehensive multiple regression analysis for predicting continuous outcomes using multiple predictors. Developed for the Strategic Exercise Information and Research unit in Likes Institute, at JAMK University of Applied Sciences, this module provides detailed model diagnostics, coefficient analysis, and comprehensive regression statistics.

## Features

- **Complete Regression Analysis**: Multiple regression with comprehensive diagnostics
- **Model Validation**: R-squared, adjusted R-squared, RMSE, and MAE calculations
- **Coefficient Analysis**: Detailed coefficient statistics with confidence intervals
- **Advanced Diagnostics**: VIF analysis, residual plots, and Q-Q plots
- **Excel Integration**: Multi-sheet results with detailed statistics and predictions
- **Terminal Feedback**: Comprehensive model summary and coefficient analysis
- **Assumption Testing**: Automated tests for regression assumptions
- **Tested**: Functioning tested with simulated data

## Hardware Requirements

- **Python:** 3.8 or higher
- **RAM:** 8GB recommended
- **Storage:** 1GB free space for analysis outputs
- **OS:** Windows 10/11, MacOS, or Linux
- **CPU:** Multi-core recommended for larger datasets

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/juhanimerilehto/multiple-regression-analysis-script.git
cd multiple-regression-analysis-script
```

### 2. Create a virtual environment:
```bash
python -m venv stats-env
source stats-env/bin/activate  # For Windows: stats-env\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python multipleregression.py
```

With custom parameters:
```bash
python multipleregression.py --excel_path "your_data.xlsx" --target "Target" --features "Feature1,Feature2,Feature3"
```

## Configuration Parameters

- `excel_path`: Path to Excel file (default: 'data.xlsx')
- `feature_columns`: List of predictor variables (default: ['Feature1', 'Feature2', 'Feature3'])
- `target_column`: Outcome variable (default: 'Target')
- `test_size`: Proportion of test data (default: 0.2)
- `random_state`: Random seed for reproducibility (default: 42)
- `output_prefix`: Prefix for output files (default: 'multiple_regression')

## File Structure

```plaintext
├── multiple-regression-analysis-script/
│   ├── assets/
│   │   └── likes_str_logo.png
│   ├── multipleregression.py
│   ├── requirements.txt
│   └── README.md
```

## Credits

- **Juhani Merilehto (@juhanimerilehto)** – Specialist, Data and Statistics
- **JAMK Likes** – Organization sponsor

## License

This project is licensed for free use under the condition that proper credit is given to Juhani Merilehto (@juhanimerilehto) and JAMK Likes institute. You are free to use, modify, and distribute this project, provided that you mention the original author and institution and do not hold them liable for any consequences arising from the use of the software.