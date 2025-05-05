# Task 7: Support Vector Machines (SVM) - AI & ML Internship

## Objective
Use SVMs for linear and non-linear classification on a binary dataset.

## Dataset
We used the built-in Breast Cancer dataset from `sklearn.datasets`.

## Libraries Used
- scikit-learn
- NumPy
- Matplotlib

## Concepts Applied
- Support Vector Machines (SVM)
- Linear and RBF Kernels
- PCA for 2D Visualization
- Cross-Validation
- Hyperparameter Tuning (GridSearchCV)

## 🛠Tasks Performed
1. Loaded and scaled the breast cancer dataset.
2. Reduced dimensionality to 2D using PCA for visualization.
3. Trained two SVM models:
   - Linear kernel
   - RBF kernel
4. Evaluated both models using accuracy, confusion matrix, and classification report.
5. Performed cross-validation.
6. Tuned hyperparameters for RBF using GridSearchCV.
7. Visualized decision boundaries.

## Results
- Linear SVM accuracy: ~94-96%
- RBF SVM accuracy: ~93-96%
- Best hyperparameters (RBF): `{'C': 10, 'gamma': 0.01}` (example result)

## How to Run
```bash
Linear Kernel SVM
[[42  1]
 [ 0 71]]
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        43
           1       0.99      1.00      0.99        71

    accuracy                           0.99       114
   macro avg       0.99      0.99      0.99       114
weighted avg       0.99      0.99      0.99       114

Accuracy: 0.9912280701754386

RBF Kernel SVM
[[39  4]
 [ 0 71]]
              precision    recall  f1-score   support

           0       1.00      0.91      0.95        43
           1       0.95      1.00      0.97        71

    accuracy                           0.96       114
   macro avg       0.97      0.95      0.96       114
weighted avg       0.97      0.96      0.96       114

Accuracy: 0.9649122807017544

Cross-validation Accuracy (Linear): 0.9525850023288308
Cross-validation Accuracy (RBF): 0.9402732494954199

Best Hyperparameters for RBF SVM: {'C': 10, 'gamma': 0.1}
Best Score: 0.9437820214252446
