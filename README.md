# Predicting Bioprinting Properties for 3D Printing Bioinks

This repository contains resources, data, and code for predicting bioprinting properties of bioinks used in 3D printing. By analyzing a comprehensive database of over 1200 bioprinting tests, this project aims to optimize bioprinting conditions and materials through machine learning (ML) and artificial intelligence (AI)-based models.

The best model developed in this project leverages **perturbation theory-based operators**, achieving high specificity and sensitivity in both training and external validation datasets. The performance of the proposed model has been compared with neural networks, yielding very similar results. The tool provides an accessible way to predict properties for *in silico* bioprinting assays, streamlining optimization and reducing experimental steps.

---

## Objectives

- Create a comprehensive database of experimental bioprinting conditions and results.
- Train and validate AI/ML-based models to predict scaffold properties and optimize printing parameters.
- Develop tools to forecast results for new experiments and aid in the design of innovative bioinks.
- Compare perturbation theory-based models with neural network approaches to validate effectiveness.

---

## Repository Contents

### Folders
- **`references/`**  
  Collection of articles used to compile the database.

- **`results_code/`**  
  Code used to generate and analyze the prediction results.

### Scripts
- **`1_1_MA_alva.py`**  
  Script to calculate moving averages for data descriptors.

- **`1_1_MA_alva_optimized.py`**  
  Optimized version of `1_1_MA_alva.py` to speed up processing by using subtables.

- **`1_2_MA_dragon.py`**  
  Similar to `1_1_MA_alva.py`, but for descriptors calculated using the Dragon software.

- **`3_Models.py`**  
  Code for training and comparing different models.

- **`3_dataprep.py`**  
  Prepares new datasets for prediction by trained models.

### Pre-trained Models
- **`DTC_best.joblib`**  
  Pre-trained Decision Tree Classifier (DTC) model.

- **`LDA_best.joblib`**  
  Pre-trained Linear Discriminant Analysis (LDA) model.

### Supplementary Files
- **`S1.xlsx`**  
  Supplementary material S1: The full database used in the study.

- **`S2.pdf`**  
  Supplementary material S2: Decision Tree Classifier (DTC) diagram, including node information.

### Results
- **`results.xlsx`**  
  Predicted results from the trained models.

---

## Getting Started

### Prerequisites
- Python 3.7+  
- Required libraries: `scikit-learn`, `pandas`, `joblib`, etc. (Refer to the individual scripts for additional dependencies.)

### Running the Code
1. **Calculate Moving Averages:**  
   Run `1_1_MA_alva.py` or `1_2_MA_dragon.py` to calculate moving averages based on dataset descriptors.

2. **Optimize Dataset Processing:**  
   Use `1_1_MA_alva_optimized.py` to improve processing speed for large datasets.

3. **Train Models:**  
   Execute `3_Models.py` to train new ML models or validate existing ones.

4. **Prepare New Data for Prediction:**  
   Use `3_dataprep.py` to process new experimental data before prediction.

5. **Make Predictions:**  
   Load the best models (`DTC_best.joblib` or `LDA_best.joblib`) and predict scaffold properties for *in silico* bioprinting.

---

## Database and Supplementary Material

- **Database**: The dataset (`S1.xlsx`) contains over 1200 bioprinting tests covering various conditions such as print pressure, temperature, and needle parameters.
- **Decision Tree Classifier (DTC)**: The DTC diagram (`S2.pdf`) provides detailed information on the classification process.

---

## Citation

If you use this repository in your research, please cite the corresponding publication. Supplementary materials can be found in `S1.xlsx` and `S2.pdf`.

---
