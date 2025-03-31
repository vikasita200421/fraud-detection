# Fraud Detection Project

## Overview
This project aims to detect fraudulent transactions using machine learning. The dataset contains transaction details, and the model is trained to classify transactions as fraudulent or legitimate.

## Features
- **Data Preprocessing:** Cleans and prepares the dataset for training.
- **Machine Learning Model:** Uses a trained model to classify transactions.
- **Model Persistence:** Saves and loads trained models using `pickle`.
- **Web API:** Exposes endpoints for fraud detection using `Flask`.

## Project Structure
```
fraud-detection/
│── data/                     # Contains large dataset files (not in repo)
│── models/                   # Stores trained models
│── app.py                    # Flask API for fraud detection
│── preprocess.py              # Data preprocessing script
│── train.py                   # Model training script
│── requirements.txt           # Python dependencies
│── README.md                  # Project documentation
│── .gitignore                 # Ignoring large files
```

## Setup and Installation
### Prerequisites
- Python 3.x
- Git
- Virtual environment (optional but recommended)

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Training the Model
Run the training script to preprocess data and train the model:
```sh
python train.py
```

### Running the API
Start the Flask server to make predictions:
```sh
python app.py
```
The API will be available at `http://127.0.0.1:5000`.

## API Usage
### Endpoint: Predict Fraud
**POST** `/predict`
#### Request Body:
```json
{
    "amount": 1200.50,
    "time": 24500,
    "transaction_type": "online"
}
```
#### Response:
```json
{
    "fraud": true,
    "probability": 0.92
}
```

## Handling Large Files
- **Large files like `creditcard.csv`, `fraud_data.csv`, and `preprocessed_data.pkl` are stored in `data/` (ignored in `.gitignore`).**
- If needed, download the dataset manually and place it inside the `data/` folder.

## Future Enhancements
- Improve model accuracy with deep learning.
- Deploy on cloud services (AWS, GCP, Azure).
- Implement real-time fraud detection.

