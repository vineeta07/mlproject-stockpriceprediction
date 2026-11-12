"""Stock Price Analysis Dashboard.

Loads the trained LSTM model (saved_model.keras) and shows actual vs.
predicted closing prices for NSE Tata Global Beverages, plus an optional
tab comparing other stocks from stock_data.csv.

The model is run with plain NumPy (weights read straight from the .keras
file), so the website does not need TensorFlow. This keeps memory low
enough for free hosting plans such as Render's.

Run locally:   python app.py
Run on Render: gunicorn app:server
"""

import os
import zipfile
from io import BytesIO

import dash
import h5py
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler

NSE_CSV = "NSE-Tata-Global-Beverages-Limited.csv"
STOCKS_CSV = "stock_data.csv"
MODEL_FILE = "saved_model.keras"
TRAIN_SIZE = 987
WINDOW = 60

app = dash.Dash(__name__, title="Stock Price Prediction")
server = app.server  # used by gunicorn


# ---------- Lightweight LSTM inference (no TensorFlow needed) ----------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTMModel:
    """Runs the notebook's LSTM(50) -> LSTM(50) -> Dense(1) model in NumPy."""

    def __init__(self, path):
        with zipfile.ZipFile(path) as z:
            weights = BytesIO(z.read("model.weights.h5"))
        with h5py.File(weights, "r") as f:
            layers = f["layers"]
            lstm_names = sorted(n for n in layers if "cell" in layers[n])
            dense_name = next(n for n in layers if n.startswith("dense"))
            self.lstms = [tuple(np.array(layers[n]["cell"]["vars"][str(i)]) for i in range(3))
                          for n in lstm_names]
            self.dense = tuple(np.array(layers[dense_name]["vars"][str(i)]) for i in range(2))

    @staticmethod
    def _lstm(x, kernel, recurrent, bias, return_sequences):
        batch, steps, _ = x.shape
        units = recurrent.shape[0]
        h = np.zeros((batch, units))
        c = np.zeros((batch, units))
        outputs = []
        for t in range(steps):
            z = x[:, t, :] @ kernel + h @ recurrent + bias
            i, f, g, o = np.split(z, 4, axis=1)  # Keras gate order: i, f, c, o
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
            h = _sigmoid(o) * np.tanh(c)
            outputs.append(h)
        return np.stack(outputs, axis=1) if return_sequences else h

    def predict(self, x):
        out = np.asarray(x, dtype=np.float64)
        for n, (k, r, b) in enumerate(self.lstms):
            out = self._lstm(out, k, r, b, return_sequences=n < len(self.lstms) - 1)
        return out @ self.dense[0] + self.dense[1]


# ---------- LSTM predictions on the NSE data ----------
df_nse = pd.read_csv(NSE_CSV)
df_nse["Date"] = pd.to_datetime(df_nse["Date"], format="%Y-%m-%d")
new_data = df_nse.sort_values("Date")[["Date", "Close"]].set_index("Date")

dataset = new_data[["Close"]].values.astype(float)
valid_len = len(dataset) - TRAIN_SIZE

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(dataset)

model = NumpyLSTMModel(MODEL_FILE)

inputs = scaler.transform(dataset[len(dataset) - valid_len - WINDOW:])
X_test = np.array([inputs[i - WINDOW:i, 0] for i in range(WINDOW, len(inputs))])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
closing_price = scaler.inverse_transform(model.predict(X_test))

train = new_data[:TRAIN_SIZE]
valid = new_data[TRAIN_SIZE:].copy()
valid["Predictions"] = closing_price.flatten()


# ---------- Optional multi-stock data ----------
df_stocks = pd.read_csv(STOCKS_CSV) if os.path.exists(STOCKS_CSV) else None
STOCK_NAMES = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft"}
STOCK_OPTIONS = [{"label": v, "value": k} for k, v in STOCK_NAMES.items()]
COLORWAY = ["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"]
RANGE_X_AXIS = {
    "title": "Date",
    "rangeselector": {"buttons": [
        {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
        {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
        {"step": "all"},
    ]},
    "rangeslider": {"visible": True},
    "type": "date",
}
DROPDOWN_STYLE = {"display": "block", "margin-left": "auto",
                  "margin-right": "auto", "width": "60%"}


def comparison_figure():
    return {
        "data": [
            go.Scatter(x=train.index, y=train["Close"], mode="lines", name="Training data"),
            go.Scatter(x=valid.index, y=valid["Close"], mode="lines", name="Actual price"),
            go.Scatter(x=valid.index, y=valid["Predictions"], mode="lines", name="LSTM prediction"),
        ],
        "layout": go.Layout(title="Actual vs. LSTM Predicted Closing Price",
                            xaxis={"title": "Date"}, yaxis={"title": "Closing Price (INR)"},
                            height=550),
    }


def scatter_figure(y, title):
    return {
        "data": [go.Scatter(x=valid.index, y=y, mode="markers")],
        "layout": go.Layout(title=title, xaxis={"title": "Date"},
                            yaxis={"title": "Closing Rate"}),
    }


tabs = [
    dcc.Tab(label="NSE-TATAGLOBAL Stock Data", children=[html.Div([
        dcc.Graph(id="comparison", figure=comparison_figure()),
        html.H2("Actual closing price", style={"textAlign": "center"}),
        dcc.Graph(id="actual-data", figure=scatter_figure(valid["Close"], "Actual")),
        html.H2("LSTM predicted closing price", style={"textAlign": "center"}),
        dcc.Graph(id="predicted-data", figure=scatter_figure(valid["Predictions"], "Predicted")),
    ])]),
]

if df_stocks is not None:
    tabs.append(dcc.Tab(label="Other Stocks", children=[html.Div([
        html.H1("Stocks High vs Lows", style={"textAlign": "center"}),
        dcc.Dropdown(id="my-dropdown", options=STOCK_OPTIONS, multi=True,
                     value=["FB"], style=DROPDOWN_STYLE),
        dcc.Graph(id="highlow"),
        html.H1("Market Volume", style={"textAlign": "center"}),
        dcc.Dropdown(id="my-dropdown2", options=STOCK_OPTIONS, multi=True,
                     value=["FB"], style=DROPDOWN_STYLE),
        dcc.Graph(id="volume"),
    ], className="container")]))

app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    html.P("Stock price prediction using an LSTM neural network",
           style={"textAlign": "center", "color": "#555"}),
    dcc.Tabs(id="tabs", children=tabs),
])


def _stock_rows(stock):
    return df_stocks[df_stocks["Stock"] == stock]


def _names(selected):
    return ", ".join(STOCK_NAMES[s] for s in selected)


if df_stocks is not None:
    @app.callback(Output("highlow", "figure"), [Input("my-dropdown", "value")])
    def update_highlow(selected):
        selected = selected or []
        data = []
        for col, opacity in (("High", 0.7), ("Low", 0.6)):
            for stock in selected:
                rows = _stock_rows(stock)
                data.append(go.Scatter(x=rows["Date"], y=rows[col], mode="lines",
                                       opacity=opacity, name=f"{col} {STOCK_NAMES[stock]}"))
        return {"data": data, "layout": go.Layout(
            colorway=COLORWAY, height=600,
            title=f"High and Low Prices for {_names(selected)} Over Time",
            xaxis=RANGE_X_AXIS, yaxis={"title": "Price (USD)"})}

    @app.callback(Output("volume", "figure"), [Input("my-dropdown2", "value")])
    def update_volume(selected):
        selected = selected or []
        data = [go.Scatter(x=_stock_rows(s)["Date"], y=_stock_rows(s)["Volume"],
                           mode="lines", opacity=0.7, name=f"Volume {STOCK_NAMES[s]}")
                for s in selected]
        return {"data": data, "layout": go.Layout(
            colorway=COLORWAY, height=600,
            title=f"Market Volume for {_names(selected)} Over Time",
            xaxis=RANGE_X_AXIS, yaxis={"title": "Transactions Volume"})}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
