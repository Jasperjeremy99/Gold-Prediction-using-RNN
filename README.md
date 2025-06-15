# Gold-Price-Prediction-RNN

This repository contains a time series forecasting project focused on predicting gold prices in USD. The project utilizes Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, to model and forecast historical gold price data across various currencies.

## Project Goal

The primary objective of this project is to develop a robust and accurate machine learning model for predicting gold prices in USD. The aim is to create a model that is not prone to overfitting and provides reliable forecasts when applied to new, real-world data.

## Files Overview

* **`gold_price_prediction_time_series.ipynb`**: This Jupyter notebook encompasses the entire project pipeline:
    * **Data Preprocessing**: Loading and initial exploration of historical gold price data from 1985 to 2023, ensuring data quality and preparing it for time series analysis.
    * **Predictive Modeling**: Implementation and training of various RNN/LSTM architectures for time series forecasting.
    * **Experiments Report**: Detailed evaluation of different RNN models, comparing their performance using Mean Absolute Error (MAE) across multiple forecasting horizons.
    * **Model Selection**: Identification of the "RNN 5" model as the best performer due to its consistent low MAE values, indicating its strong generalization capabilities.
    * **Future Improvements**: Recommendations for model optimization, including hyperparameter tuning, exploring advanced architectures (e.g., Bi-directional LSTMs, CNN-LSTM), and implementing early stopping techniques.

<!--* **`Part3_GoldPrice.csv`**: This CSV file contains the historical gold price data from 1985 to 2023, including prices in USD, EUR, GBP, INR, AED, and CNY. This dataset is crucial for the execution and reproducibility of the Jupyter notebook.

## Dataset

The project relies on the `Part3_GoldPrice.csv` dataset, which includes daily gold prices from 1985 to 2023 across several major currencies. This dataset is read directly from a specified path within the notebook.-->

## Technologies Used

* Python
* Pandas (for data manipulation)
* NumPy (for numerical operations)
* Matplotlib (for data visualization)
* Scikit-learn (for data preprocessing like `MinMaxScaler`)
* TensorFlow/Keras (for building and training the RNN/LSTM models)

## Getting Started

To explore or run the analysis:

1.  Clone this repository.
2.  Ensure you have Python installed along with the necessary libraries (you can install them via `pip install pandas numpy matplotlib scikit-learn tensorflow`).
3.  Open `gold_price_prediction_time_series.ipynb` in a Jupyter environment (e.g., Jupyter Lab, VS Code with Jupyter extension, or Google Colab).
4.  Run the cells sequentially to execute the data loading, preprocessing, model training, and evaluation steps.

## Contact

For any questions or further information, please contact [Jeremiyah/jeremypeter016@gmail.com].
