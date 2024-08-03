# Outlier Detection in Open Food Facts with Random Cut Forest

This repository contains a Python project that uses the Random Cut Forest algorithm to identify potential outliers in the Open Food Facts dataset, specifically focusing on products with unusual Nutri-Scores.

## Project Overview
The goal of this project is to leverage machine learning to highlight potential inconsistencies or errors in the nutritional information of food products listed on Open Food Facts. By identifying products with Nutri-Scores that deviate significantly from similar products, we aim to:

Improve data quality: Flag potential errors in nutrient declarations or Nutri-Score calculations.
Identify potential mislabeling: Highlight products where the stated nutritional content seems inconsistent with its category or ingredients.
Support Open Food Facts contributors: Provide insights to guide further investigation and data verification efforts.
## Dataset
The project utilizes the Open Food Facts dataset, a free and open database of food products from around the world. The dataset contains information on nutritional values, ingredients, allergens, additives, and more. You can access and download the dataset from https://world.openfoodfacts.org/data.

## Methodology
- Data Preprocessing: 
The raw Open Food Facts dataset is preprocessed to:
Handle missing values.
Select relevant features for outlier detection (e.g., energy, fat, sugar, fiber, protein, salt, Nutri-Score).
Normalize numerical features.
- Random Cut Forest Model: We employ the Random Cut Forest algorithm, an unsupervised anomaly detection technique, to identify outliers. This algorithm is well-suited for high-dimensional data and can efficiently handle large datasets.
Outlier Identification: Products with anomaly scores above a certain threshold are flagged as potential outliers.
Analysis and Visualization: The identified outliers are analyzed to understand the reasons behind their unusual Nutri-Scores. Visualizations are used to explore the distribution of outliers and their characteristics.

## Repository Structure
data/: Directory to store the Open Food Facts dataset.
notebooks/: Jupyter notebooks for data exploration, model training, and outlier analysis.
src/: Python scripts containing data preprocessing, model implementation, and visualization functions.
models/: Directory to save trained models.
results/: Directory to store analysis results and visualizations.
README.md: This file, providing an overview of the project.

## Getting Started
Clone the repository: git clone https://github.com/baslia/nutri_score.git
Install dependencies: pip install -r requirements.txt
Download the Open Food Facts dataset and place it in the data/ directory.
Run the Jupyter notebooks in the notebooks/ directory to explore the data, train the model, and analyze the results.
Contributing
Contributions to this project are welcome! If you find any issues, have suggestions for improvements, or want to contribute new features, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.