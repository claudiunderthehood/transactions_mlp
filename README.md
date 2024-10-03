
---

# Detecting Credit Card Frauds with MLP

This project focuses on detecting fraudulent activities using machine learning techniques, specifically a Multilayer Perceptron (MLP) model. The dataset used for training and testing is generated and transformed to enhance the model’s performance. This project applies feature generation and transformation techniques to prepare the data for the MLP, which is then used to classify fraudulent and non-fraudulent transactions.

## Project Overview

The repository contains the following core components:
- **Data Generation**: A custom dataset is generated for fraud detection purposes.
- **Feature Engineering**: Various transformation techniques are applied to enhance the dataset's predictive power.
- **Modeling**: The project employs an MLP as the primary classifier to detect fraudulent transactions.

## Folder Structure
- `notebooks/`: Contains Jupyter notebooks used for feature engineering and data transformation.
- `src/`: Includes Python scripts for data generation and model training.
- `data/`: Stores generated datasets (not included in the repository).
- `models/`: Directory for saving the trained models.
- `classification_modules/`: Directory with all the related classification methods.
- `generator_modules/`: Contains all the tools to generate the base dataset.
- `deepLearning_modules/`: All the modules for training and preparing the data for the Neural Networks.

## Setup and Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:
   Use the provided `requirements.txt` file to install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Generate the Data

The data for the project is created using a custom data generator. To generate the dataset, navigate to the `src` folder and run the `generator.py` script:

```bash
cd src
python generator.py
```

This script will generate a dataset of transactions with labels indicating whether each transaction is fraudulent or not.

### Step 2: Feature Transformation

Once the data has been generated, you can transform the features using the `feature_engineering` notebook located in the `notebooks` folder. The notebook applies several transformation techniques, such as one-hot encoding, standardization, and polynomial feature generation.

To transform the data:
1. Open the `feature_engineering.ipynb` notebook in your preferred Jupyter environment.
2. Execute the cells to apply the transformations and prepare the dataset for training.

### Step 3: Train the MLP Model

Once the dataset has been prepared, you can train the MLP model by running the relevant notebook or script in the repository. The MLP will be trained using the transformed features and evaluated based on metrics such as accuracy, precision, and recall.

## Citation and Licensing

The idea behind the codebase for the data generator and feature transformation is inspired by the **Machine Learning Group (Université Libre de Bruxelles - ULB).** You can find the original source code at the following link:  
[Transaction Generator Handbook - Chapter 3](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook/tree/main/Chapter_3_GettingStarted)