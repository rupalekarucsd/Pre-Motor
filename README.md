# Pre-Motor: Neural Decoding MVP

This project is a Brain-Computer Interface (BCI) dashboard. It uses machine learning to look at EEG brain waves and predict whether a subject was imagining moving their **Left Hand** or **Right Hand**.

## What's in this project?
* **[app.py](cci:7://file:///Users/rohanupalekar/Desktop/Projects/Pre-Motor/app.py:0:0-0:0)**: The main visual dashboard. Run this to see the UI, view the brain waves, and test the model's accuracy on the data.
* **[data_ingestion.py](cci:7://file:///Users/rohanupalekar/Desktop/Projects/Pre-Motor/data_ingestion.py:0:0-0:0)**: The script used to download the raw brain wave data from the MOABB database, filter it, and train the machine learning pipeline. 
* **[csp_lda_active_model.pkl](cci:7://file:///Users/rohanupalekar/Desktop/Projects/Pre-Motor/csp_lda_active_model.pkl:0:0-0:0)**: The saved mathematical "brain" (the trained model weights) so the dashboard doesn't have to retrain every time you open it.
* **[generate_matrix.py](cci:7://file:///Users/rohanupalekar/Desktop/Projects/Pre-Motor/generate_matrix.py:0:0-0:0)**: A testing script that manually runs an 80/20 train/test split to generate a Confusion Matrix, proving the model's accuracy.

## How to run the Dashboard

### 1. Install the required libraries
Open your computer's terminal, navigate to this folder, and run:
```bash
pip install -r requirements.txt
python -m streamlit run app.py
