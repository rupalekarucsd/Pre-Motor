import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate the confusion matrix for Model A (CSP + LDA)
# Ingestion (Active Phase)
dataset = BNCI2014_001()
dataset.subject_list = [1]
paradigm = MotorImagery(n_classes=2, events=['left_hand', 'right_hand'], fmin=8, fmax=32, tmin=0.5, tmax=2.5)
X, labels, _ = paradigm.get_data(dataset=dataset, subjects=[1])

# Train/ Test/ Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# Train Model A
model_a = Pipeline([
    ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("scaler", StandardScaler()),
    ("lda", LinearDiscriminantAnalysis())
])
model_a.fit(X_train, y_train)

# Predict and plot
y_pred = model_a.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=['left_hand', 'right_hand'])

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Left Hand', 'Right Hand'])
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix: Model A (CSP + LDA)\nActive Motor Imagery Phase", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()