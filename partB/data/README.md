# Data — Iris Binary Classification

## Dataset Source

This project uses the **Iris dataset** built into scikit-learn. No external download is required.

## How to Load

The dataset is loaded inside each notebook using:

```python
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
mask = iris.target < 2                
X_raw = iris.data[mask][:, 2:4]      
y = np.where(iris.target[mask] == 0, -1, 1)  
```

## Preprocessing

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)   
```

The scaler is fit only on the training set and then applied to the test set.

## How It Is Used

| Task Notebook | Use |
|--------------|-----|
| task_2_1.ipynb | Load, preprocess, visualize, justify dataset choice |
| task_2_2.ipynb | Train offset-free SVM and baseline SVC |
| task_2_3.ipynb | Evaluate and report results |
| task_3_1.ipynb | Ablation experiments over same dataset |
| task_3_2.ipynb | Failure mode: uses a **synthetic** non-centered dataset (no external file needed) |

## Subset Used

- **Samples:** 100 (Setosa: 50, Versicolor: 50)
- **Features:** Petal Length (cm), Petal Width (cm)
- **Task:** Binary classification
- **Split:** 80 train / 20 test, stratified, random_state=42
