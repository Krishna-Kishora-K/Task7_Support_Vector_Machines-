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
- Linear SVM accuracy: ~96-97%
- RBF SVM accuracy: ~97-98%
- Best hyperparameters (RBF): `{'C': 10, 'gamma': 0.01}` (example result)

## How to Run
```bash
pip install numpy matplotlib scikit-learn
python svm_task7.py
