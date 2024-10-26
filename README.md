# OpenPriceNVIDIAForecasting



### NVIDIA Stock Price Prediction using Attention-based CNN

This project aims to predict the opening price of NVIDIA stock by utilizing a custom Attention-based Convolutional Neural Network (CNN) architecture. By leveraging feature engineering techniques and incorporating time-series features, the model strives to capture complex temporal patterns in the stock price data.


### Introduction
Accurate stock price prediction is critical in financial markets, as it provides valuable insights for making profitable investment decisions. This project applies deep learning techniques, particularly a hybrid model incorporating CNN layers with Attention mechanisms, to predict the daily opening prices of NVIDIA stock.

### Dataset
The dataset used in this project includes stock market data sourced from a reliable financial API. It contains the following key features:

Date: The trading date in YYYY-MM-DD format.
Symbol: The ticker symbol for NVIDIA (NVDA).
Open: The opening price of the stock on that trading day.
High: The highest price reached during the trading session.
Low: The lowest price reached during the trading session.
Close: The closing price of the stock at the end of the trading day.
Volume: The total number of shares traded during that day.

### link to the dataset --> https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks?select=sp500_stocks.csv

### Feature Engineering
Feature engineering is crucial in making temporal models robust for time-series forecasting. The following features were created:

Moving Averages (MA_5, MA_10): Averages over 5 and 10 days, smoothing out price fluctuations.
Lag Features (Lag_1, Lag_2): Previous day(s) price data, enabling the model to use past information.
This engineered dataset is then split into training and testing sets.

### Model Architecture
The architecture incorporates Attention layers within a deep Convolutional Neural Network to capture intricate dependencies in stock prices. The model includes:
Multiple Conv1D layers to extract features from temporal sequences.
Layer Normalization after each Conv1D block to stabilize training.
An Attention mechanism that helps in weighting critical features.
A fully connected Dense layer at the end to output the predicted opening price.
This architecture is designed to capture both long-term and short-term dependencies, making it suitable for volatile stock price data.

### Model Training and Evaluation
The model was trained using Mean Squared Error (MSE) loss and Mean Absolute Error (MAE) as the metric. Early stopping was applied to prevent overfitting, and the model was optimized using the Adam optimizer with a learning rate of 0.001.

Epoch 1/250: loss - 26200.8945, MAE - 92.3910, val_loss - 28515.2168
Epoch 10/250: loss - 16108.6318, MAE - 58.3353, val_loss - 16697.6758
Epoch 40/250: loss - 2924.0425, MAE - 24.6177, val_loss - 1760.6752

### Dependencies
The following libraries are required to run the project:
pandas: Data manipulation
scikit-learn: Train-test split
tensorflow: Model building and training

### Usage
Preprocess the Data: Run the feature engineering code to prepare the dataset.
Train the Model: Execute the model training code with the provided architecture and callbacks.
Evaluate: Track the validation loss and MAE over epochs for model performance analysis.
