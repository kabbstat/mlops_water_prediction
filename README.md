# Water Potability Prediction using Machine Learning

A comprehensive MLOps project for predicting water potability using machine learning techniques. This project follows industry best practices with a well-structured data science workflow and reproducible analysis pipeline.

## 🎯 Project Overview

This project aims to predict water potability (whether water is safe for drinking) using various water quality parameters. The model helps in assessing water quality and ensuring safe drinking water standards.

## 📊 Features

- **Data Processing Pipeline**: Automated data cleaning, transformation, and feature engineering
- **Machine Learning Models**: Multiple ML algorithms for water potability prediction
- **Model Evaluation**: Comprehensive model performance analysis and comparison
- **Visualization**: Interactive charts and graphs for data exploration and results
- **Reproducible Workflow**: Standardized project structure following data science best practices

## 🛠️ Technologies Used

- **Python**: Main programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebooks**: Interactive development and analysis

## 📁 Project Structure

```
mlops_water_prediction/
├── LICENSE
├── Makefile                 # Commands for project automation
├── README.md               # Project documentation
├── data/
│   ├── external/           # Third-party data sources
│   ├── interim/            # Intermediate processed data
│   ├── processed/          # Final datasets for modeling
│   └── raw/                # Original raw data
├── docs/                   # Documentation (Sphinx)
├── models/                 # Trained models and predictions
├── notebooks/              # Jupyter notebooks for analysis
├── references/             # Data dictionaries and manuals
├── reports/                # Generated analysis reports
│   └── figures/            # Charts and visualizations
├── requirements.txt        # Python dependencies
├── setup.py               # Package installation setup
├── src/                   # Source code
│   ├── data/              # Data processing scripts
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and prediction
│   └── visualization/     # Visualization scripts
└── tox.ini                # Testing configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kabbstat/mlops_water_prediction.git
   cd mlops_water_prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the project package**
   ```bash
   pip install -e .
   ```

## 📈 Usage

### Quick Start

1. **Data Preparation**
   ```bash
   make data
   ```

2. **Train Models**
   ```bash
   make train
   ```

3. **Generate Predictions**
   ```bash
   python src/models/predict_model.py
   ```

### Using Jupyter Notebooks

Launch Jupyter and explore the analysis notebooks:
```bash
jupyter notebook notebooks/
```

### Command Line Interface

The project includes a Makefile for common tasks:
- `make data`: Download and process data
- `make train`: Train machine learning models
- `make predict`: Generate predictions
- `make visualize`: Create visualizations
- `make clean`: Clean temporary files

## 📊 Data

The project uses water quality datasets containing various parameters such as:
- pH levels
- Hardness
- Solids (Total Dissolved Solids)
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity
- Potability (target variable)

## 🤖 Models

The project implements and compares multiple machine learning algorithms:
- Logistic Regression
- Random Forest
- Support Vector Machine
- Gradient Boosting
- Neural Networks

## 📋 Results

Model performance metrics and comparisons are available in the `reports/` directory, including:
- Accuracy scores
- Precision, Recall, F1-score
- ROC curves
- Feature importance analysis
- Model comparison charts

## 🔄 MLOps Pipeline

The project follows MLOps best practices:
- **Version Control**: Git for code and data versioning
- **Reproducibility**: Standardized project structure and requirements
- **Automation**: Makefile for common tasks
- **Documentation**: Comprehensive code and project documentation
- **Testing**: Automated testing with tox
- **Monitoring**: Model performance tracking

## 📚 Documentation

Detailed documentation is available in the `docs/` directory. To build the documentation:
```bash
cd docs/
make html
```

## 🧪 Testing

Run tests using tox:
```bash
tox
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 👨‍💻 Author

**kabbaj mohamed stat**
- GitHub: [@kabbstat](https://github.com/kabbstat)

## 🙏 Acknowledgments

- Project structure based on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template
- Thanks to the open-source community for the tools and libraries used

## 📞 Support

If you have any questions or need help with the project, please:
1. Check the documentation in the `docs/` directory
2. Look through existing issues on GitHub
3. Create a new issue if your question isn't answered

---

*This project demonstrates MLOps best practices for water quality prediction, combining data science workflows with engineering principles for reproducible and scalable machine learning solutions.*
