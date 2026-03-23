import warnings
warnings.filterwarnings("ignore")

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import joblib
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Data Ingestion (Active Phase) 
print("Initializing MOABB and connecting to BCI Competition IV-2a...")
dataset = BNCI2014_001()
dataset.subject_list = [1]

# Shifting the time window to 0.5s - 2.5s to capture peak motor imagery
paradigm = MotorImagery(n_classes=2, events=['left_hand', 'right_hand'], fmin=8, fmax=32, tmin=0.5, tmax=2.5)

print("Downloading and extracting epochs...")
X, labels, _ = paradigm.get_data(dataset=dataset, subjects=[1])
print(f"Extraction complete: {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} timepoints.")

 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model A: Cleaned (CSP and LDA) Data
print("\nTraining Model A: CSP + LDA (The Active Pipeline)...")
model_a = Pipeline([
    ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("scaler", StandardScaler()),
    ("lda", LinearDiscriminantAnalysis())
])
scores_a = cross_val_score(model_a, X, labels, cv=cv, scoring='accuracy')
accuracy_a = np.mean(scores_a) * 100
print(f"-> Model A Accuracy: {accuracy_a:.2f}%")

# Model B: The Raw Data Baseline
print("\nTraining Model B: Raw Data + LDA (The Unfiltered Baseline)...")
X_raw = X.reshape(X.shape[0], -1) 
model_b = LinearDiscriminantAnalysis()
scores_b = cross_val_score(model_b, X_raw, labels, cv=cv, scoring='accuracy')
accuracy_b = np.mean(scores_b) * 100
print(f"-> Model B Accuracy: {accuracy_b:.2f}%")

#MODEL PERSISTENCE 
print("\nTraining final pipeline on full dataset before saving...")
model_a.fit(X, labels) 

print("Saving the winning pipeline to disk...")
model_filename = "csp_lda_active_model.pkl"
joblib.dump(model_a, model_filename)
print(f"Success: Model saved as '{model_filename}'")