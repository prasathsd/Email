# **Email Campaign Optimization Model**

## **1. Project Overview**

This project involves developing a **machine learning model** aimed at optimizing **email campaign performance**. Specifically, the model predicts whether a recipient will click on an email based on various user-related and email-related features.

The dataset has a **class imbalance** where class `0` (no click) is much more frequent than class `1` (clicked). The challenge is to create a model that balances the predictions, especially for the minority class, while optimizing overall performance.

---

## **2. Problem Statement**

The dataset contains the following classes:
- **0**: Did not click
- **1**: Clicked

The objective is to predict whether a user will click on an email based on their behavior and email attributes, dealing with the challenge of imbalanced classes.

---

## **3. Data Preprocessing**

### **3.1 Data Cleaning**
- Removed unnecessary columns.
- Handled missing values using **SimpleImputer** to replace missing numeric values with the mean.
- Applied **feature scaling** for normalization.

### **3.2 Data Splitting**
The dataset is split into **training** and **testing** sets (80/20 split).

---

## **4. Model Development**

### **4.1 Handling Class Imbalance**
We applied **SMOTE** to the training data to generate synthetic instances for the minority class (`1`). This helps balance the dataset for more reliable model training.

### **4.2 Model Selection**
We used **Random Forest Classifier** with `class_weight="balanced"` to handle imbalanced data. Other models can also be explored if needed.

### **4.3 Hyperparameter Tuning**
We performed **GridSearchCV** to tune hyperparameters like `max_depth`, `min_samples_split`, and `n_estimators` to improve model performance.

---

## **5. Model Evaluation**

### **5.1 Metrics**
- **AUC (Area Under the Curve)**: The model’s ability to distinguish between the two classes.
- **Confusion Matrix**: Helps visualize the model’s true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Provides detailed metrics like precision, recall, and F1-score for both classes.

### **5.2 Results**
- **AUC**: 0.9647
- **Accuracy**: 92.16%
- **Confusion Matrix**:
[[18101 1475] [ 5 419]]

- **Precision for Class 1 (clicked)**: 0.22
- **Recall for Class 1 (clicked)**: 0.99
- **F1-score for Class 1 (clicked)**: 0.36

---

## **6. Conclusion**

The model performs well, especially with **recall** for the minority class (`1`). However, the **precision** remains low, indicating the need for further fine-tuning or threshold adjustments.

---

## **7. Future Work**

- **Threshold Adjustment**: We could experiment with different decision thresholds to improve precision.
- **Hyperparameter Optimization**: More extensive hyperparameter tuning could help enhance performance.
- **Model Ensemble**: Combining models to reduce bias and variance.

---

## **8. Requirements**

- Python 3.x
- Dependencies:
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost (if you want to try out XGBoost)

