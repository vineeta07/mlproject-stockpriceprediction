# Stock Price Prediction using LSTM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vineeta07/mlproject-stockpriceprediction/blob/main/stockpriceprediction.ipynb)

A machine learning project that predicts the closing price of a stock from its historical prices using an **LSTM (Long Short-Term Memory)** neural network. It also includes an interactive **Dash** dashboard for exploring actual vs. predicted prices.

Submitted as the Machine Learning project for the Department of Computer Science and Engineering, DTU.

- **Submitted by:** Vineeta Kumari Meena (24/A07/056)
- **Submitted to:** Mr. Gapesh Kumar

## Objective

Build a supervised learning model that learns from past stock data and predicts future closing prices.

## Dataset

The notebook uses **NSE Tata Global Beverages Limited** historical data (CSV) with these columns:

`Date`, `Open`, `High`, `Low`, `Last`, `Close`, `Total Trade Quantity`, `Turnover (Lacs)`

Only `Date` and `Close` are used for modelling.

## Workflow

1. **Load data** with Pandas and parse `Date` as the index.
2. **Visualize** the closing price history.
3. **Sort and filter** the data to the `Date` and `Close` columns.
4. **Normalize** closing prices to `[0, 1]` with `MinMaxScaler`.
5. **Create sequences:** each sample uses the previous 60 days to predict the next day. The first 987 rows are for training and the rest for validation.
6. **Build and train the LSTM:**
   - LSTM (50 units, `return_sequences=True`)
   - LSTM (50 units)
   - Dense (1)
   - Loss: mean squared error, optimizer: Adam
7. **Predict** validation prices and inverse-transform them back to the real scale.
8. **Save** the model as `saved_model.keras`.
9. **Plot** actual vs. predicted closing prices.
10. **Dashboard:** a Dash + Plotly app that loads the saved model and shows the results interactively.

## Tech Stack

| Library      | Version |
|--------------|---------|
| Pandas       | 2.2.2   |
| NumPy        | 2.0.2   |
| Matplotlib   | 3.10.0  |
| Keras        | 3.8.0   |
| Scikit-learn | 1.6.1   |
| TensorFlow, Dash, Plotly | — |

## How to Run

1. Open the notebook in Google Colab using the badge above (or run it locally in Jupyter).
2. Upload `NSE-Tata-Global-Beverages-Limited.zip` (and optionally `stock_data.zip`) to the working directory.
3. Install dependencies if needed:

   ```bash
   pip install pandas numpy matplotlib tensorflow keras scikit-learn dash plotly
   ```

4. Run all cells in order.

## Web Dashboard

`app.py` is a Dash web app that shows actual vs. LSTM-predicted closing prices. It reads the trained weights from `saved_model.keras` and runs the model with NumPy, so the website doesn't need TensorFlow and fits on free hosting.

Files the app needs in the repo root:
- `saved_model.keras`: the trained model, created by the notebook
- `NSE-Tata-Global-Beverages-Limited.csv`: the stock data
- `stock_data.csv` (optional): enables the "Other Stocks" tab

Run it locally:

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8050.

### Deploy on Render (free)
1. Sign in at [render.com](https://render.com) with GitHub.
2. Click **New → Blueprint** and select this repository. Render reads `render.yaml` and sets everything up.
3. Click **Apply**. Your site will be live at `https://<name>.onrender.com`.

## Results

The LSTM model's predicted closing prices closely follow the actual prices on the validation set, with a low mean squared error.

## References

1. [DataFlair – Stock Price Prediction Project in Python](https://data-flair.training/blogs/stock-price-prediction-machine-learning-project-in-python/)
2. [YouTube tutorial](https://www.youtube.com/watch?v=1O_BenficgE)

## Acknowledgement

Thanks to the Machine Learning faculty for the opportunity to work on this project, and to friends and classmates for their support.
