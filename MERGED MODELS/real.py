from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor


st.set_page_config(page_title="Stock Price Prediction", layout="wide")

LOOKBACK_DAYS = 60
FORECAST_HORIZON = 100
MIN_HISTORY_DAYS = 250
STOCKS = ["AAPL", "AMZN", "GOOGL", "GOOG", "TSLA", "META", "NVDA", "V"]


@st.cache_data(ttl=3600, max_entries=32, show_spinner=False)
def load_stock_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Download daily prices; the end date selected in the UI is inclusive."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [column[0] for column in data.columns]

    if data.index.name is None:
        data.index.name = "Date"
    data = data.reset_index()
    date_column = next((column for column in data.columns if str(column).lower() in {"date", "datetime"}), None)
    if date_column is None:
        return pd.DataFrame()

    data[date_column] = pd.to_datetime(data[date_column]).dt.tz_localize(None)
    data = data.rename(columns={date_column: "Date"}).set_index("Date").sort_index()
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required_columns).issubset(data.columns):
        return pd.DataFrame()

    available_columns = required_columns + (["Adj Close"] if "Adj Close" in data.columns else [])
    data = data[available_columns].apply(pd.to_numeric, errors="coerce")
    return data.dropna(subset=["Open", "High", "Low", "Close"])


def make_market_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build features available at the close of each trading day."""
    close = data["Close"]
    features = pd.DataFrame(index=data.index)
    features["Open_Close"] = data["Open"] - close
    features["High_Low"] = data["High"] - data["Low"]
    features["Return_5d"] = close.pct_change(5)
    features["Return_20d"] = close.pct_change(20)
    features["Volatility_20d"] = close.pct_change().rolling(20).std()
    features["MA20_Ratio"] = close / close.rolling(20).mean() - 1
    features["MA50_Ratio"] = close / close.rolling(50).mean() - 1
    features["Volume_Change_20d"] = data["Volume"].pct_change(20)
    return features.replace([np.inf, -np.inf], np.nan)


def make_knn_grid_search(estimator, train_size: int, *, regression: bool) -> GridSearchCV:
    """Use only neighbor counts supported by every time-series CV fold."""
    first_fold_size = train_size // 6
    candidates = [neighbors for neighbors in (3, 5, 7, 9, 11, 13, 15) if neighbors < first_fold_size]
    if not candidates:
        candidates = [1]
    final_step_name = estimator.steps[-1][0]
    return GridSearchCV(
        estimator,
        {f"{final_step_name}__n_neighbors": candidates},
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_squared_error" if regression else "accuracy",
        n_jobs=1,
    )


@st.cache_data(ttl=86400, max_entries=8, show_spinner=False)
def train_lstm_forecast(
    ticker: str,
    close_values: tuple[float, ...],
    date_values: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Train in a clean Python process and return the chronological holdout forecast."""
    with tempfile.TemporaryDirectory(prefix="stock_lstm_") as temporary_directory:
        input_path = Path(temporary_directory) / "series.npz"
        output_path = Path(temporary_directory) / "forecast.npz"
        np.savez_compressed(
            input_path,
            close=np.asarray(close_values, dtype=np.float32),
            dates=np.asarray(date_values),
        )
        trainer_path = Path(__file__).with_name("lstm_trainer.py")
        result = subprocess.run(
            [sys.executable, str(trainer_path), str(input_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "Unknown TensorFlow training error."
            raise RuntimeError(details[-2500:])
        with np.load(output_path, allow_pickle=False) as forecast:
            actual = forecast["actual"].copy()
            predicted = forecast["predicted"].copy()
            test_dates = tuple(forecast["dates"].astype(str).tolist())
        return actual, predicted, test_dates


st.title("Stock Price Prediction")
hero_columns = st.columns((2.8, 6, 2))
hero_columns[1].image("https://miro.medium.com/max/620/0*dunTLlei47QWR7NR.gif")
intro_columns = st.columns(2)
intro_columns[0].markdown(
    "Welcome to the Stock Price Predictor. Choose a supported stock and a historical "
    "date range to view market data and model estimates."
)
intro_columns[1].markdown(
    "The dashboard compares KNN, LSTM, and Decision Tree forecasts using historical "
    "Yahoo Finance prices. Each model is evaluated separately on dates it did not train on."
)

today = date.today()
default_start = today - timedelta(days=5 * 365)
selected_stock = st.selectbox("Select stock", STOCKS)
date_columns = st.columns(2)
starting = date_columns[0].date_input("Start date", value=default_start, max_value=today)
ending = date_columns[1].date_input("End date", value=today, max_value=today)

if starting >= ending:
    st.error("The end date must be later than the start date.")
    st.stop()

try:
    with st.spinner(f"Loading {selected_stock} market data…"):
        df = load_stock_data(selected_stock, starting, ending)
except Exception as error:
    st.error(f"Could not download market data. Check your internet connection and try again. Details: {error}")
    st.stop()

if df.empty:
    st.error("Yahoo Finance returned no data for this stock and date range. Try another ticker or date range.")
    st.stop()

if len(df) < MIN_HISTORY_DAYS:
    st.warning(f"Select a range with at least {MIN_HISTORY_DAYS} trading days. This range has {len(df)}.")
    st.stop()

st.subheader("Historical data")
raw_data = df[["Open", "High", "Low", "Close", "Volume"]].head(10).reset_index()
raw_data["Date"] = raw_data["Date"].dt.strftime("%d %b %Y")
raw_data[["Open", "High", "Low", "Close"]] = raw_data[["Open", "High", "Low", "Close"]].round(2)
raw_data["Volume"] = raw_data["Volume"].fillna(0).map("{:,.0f}".format)
_, raw_data_column, _ = st.columns((2, 6, 2))
raw_data_column.dataframe(raw_data, hide_index=True, width="stretch", height=420)

st.subheader("Market trends")
chart_left, chart_right = st.columns(2)
with chart_left:
    st.markdown("**Opening price**")
    st.line_chart(df[["Open"]], height=280)
    st.markdown("**Closing price**")
    st.line_chart(df[["Close"]], height=280)
    st.markdown("**Low price**")
    st.line_chart(df[["Low"]], height=280)
with chart_right:
    st.markdown("**Daily volume**")
    st.line_chart(df[["Volume"]], height=280)
    st.markdown("**High price**")
    st.line_chart(df[["High"]], height=280)
    if "Adj Close" in df.columns:
        st.markdown("**Adjusted closing price**")
        st.line_chart(df[["Adj Close"]], height=280)

features = make_market_features(df)
next_close = df["Close"].shift(-1).rename("Actual next-day close")
next_close_values = df["Close"].shift(-1)
next_trading_date = pd.Series(df.index, index=df.index).shift(-1)
next_direction = pd.Series(
    np.where(next_close_values.isna(), np.nan, np.where(next_close_values > df["Close"], 1, -1)),
    index=df.index,
    name="Actual direction",
)

st.divider()
st.header("K-Nearest Neighbors")
classification_data = features.join(next_direction).dropna()
classification_split = int(len(classification_data) * 0.75)
if classification_split < 30 or len(classification_data) - classification_split < 1:
    st.warning("Not enough rows to evaluate the KNN classifier for this range.")
else:
    x_class = classification_data[features.columns]
    y_class = classification_data["Actual direction"]
    x_class_train, x_class_test = x_class.iloc[:classification_split], x_class.iloc[classification_split:]
    y_class_train, y_class_test = y_class.iloc[:classification_split], y_class.iloc[classification_split:]
    classifier = make_pipeline(StandardScaler(), KNeighborsClassifier())
    classifier_search = make_knn_grid_search(classifier, len(x_class_train), regression=False)
    classifier_search.fit(x_class_train, y_class_train)
    train_accuracy = accuracy_score(y_class_train, classifier_search.predict(x_class_train))
    test_predictions = classifier_search.predict(x_class_test)
    test_accuracy = accuracy_score(y_class_test, test_predictions)

    accuracy_columns = st.columns(2)
    accuracy_columns[0].metric("Train accuracy", f"{train_accuracy:.1%}")
    accuracy_columns[1].metric("Holdout accuracy", f"{test_accuracy:.1%}")

    regression_data = features.join(next_close).dropna()
    regression_split = int(len(regression_data) * 0.75)
    x_reg = regression_data[features.columns]
    y_reg = regression_data["Actual next-day close"]
    x_reg_train, x_reg_test = x_reg.iloc[:regression_split], x_reg.iloc[regression_split:]
    y_reg_train, y_reg_test = y_reg.iloc[:regression_split], y_reg.iloc[regression_split:]
    regressor = make_pipeline(StandardScaler(), KNeighborsRegressor())
    regressor_search = make_knn_grid_search(regressor, len(x_reg_train), regression=True)
    regressor_search.fit(x_reg_train, y_reg_train)
    reg_predictions = regressor_search.predict(x_reg_test)
    reg_rmse = float(np.sqrt(mean_squared_error(y_reg_test, reg_predictions)))

    classifier_table = pd.DataFrame(
        {
            "Forecast date": next_trading_date.loc[y_class_test.index].to_numpy(),
            "Actual next-day direction": y_class_test.to_numpy(),
            "Predicted direction": test_predictions,
        }
    ).head(10)
    regression_table = pd.DataFrame(
        {
            "Forecast date": next_trading_date.loc[y_reg_test.index].to_numpy(),
            "Actual next-day close": y_reg_test.to_numpy(),
            "Predicted next-day close": reg_predictions,
        }
    ).head(10)
    classifier_table["Forecast date"] = pd.to_datetime(classifier_table["Forecast date"]).dt.strftime("%d %b %Y")
    regression_table["Forecast date"] = pd.to_datetime(regression_table["Forecast date"]).dt.strftime("%d %b %Y")
    classifier_column, regression_column = st.columns(2, gap="large")
    with classifier_column:
        st.subheader("Direction classifier")
        st.dataframe(classifier_table, hide_index=True, width="stretch", height=420)
    with regression_column:
        st.subheader("Next-day close estimate")
        st.dataframe(regression_table.round(2), hide_index=True, width="stretch", height=420)
    st.metric("KNN holdout RMSE", f"${reg_rmse:,.2f}")

    fig, ax = plt.subplots(figsize=(12, 4))
    forecast_dates = next_trading_date.loc[y_reg_test.index]
    ax.plot(forecast_dates, y_reg_test.values, label="Actual next-day close")
    ax.plot(forecast_dates, reg_predictions, label="KNN estimate")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.divider()
st.header("Long Short-Term Memory (LSTM)")
with st.spinner(f"Training the LSTM for {selected_stock}… (first run can take a little while)"):
    try:
        actual_lstm, predicted_lstm, lstm_dates = train_lstm_forecast(
            selected_stock,
            tuple(df["Close"].astype(float).tolist()),
            tuple(df.index.strftime("%Y-%m-%d").tolist()),
        )
    except Exception as error:
        st.error(f"The LSTM could not be trained for this date range: {error}")
    else:
        lstm_dates_index = pd.to_datetime(lstm_dates)
        lstm_rmse = float(np.sqrt(mean_squared_error(actual_lstm, predicted_lstm)))
        lstm_mae = float(np.mean(np.abs(actual_lstm - predicted_lstm)))
        lstm_metrics = st.columns(2)
        lstm_metrics[0].metric("LSTM holdout RMSE", f"${lstm_rmse:,.2f}")
        lstm_metrics[1].metric("LSTM holdout MAE", f"${lstm_mae:,.2f}")
        lstm_chart = pd.DataFrame(
            {"Actual close": actual_lstm, "LSTM prediction": predicted_lstm},
            index=lstm_dates_index,
        )
        st.line_chart(lstm_chart, height=420)

st.divider()
st.header("Decision Tree: 100-trading-day forecast")
tree_features = features.copy()
tree_data = tree_features.copy()
tree_data["Target"] = df["Close"].shift(-FORECAST_HORIZON)
tree_data["Target position"] = np.arange(len(df)) + FORECAST_HORIZON
tree_data["Feature position"] = np.arange(len(df))
tree_data = tree_data.dropna()
split_position = int(len(df) * 0.8)
tree_train = tree_data[tree_data["Target position"] < split_position]
tree_valid = tree_data[tree_data["Target position"] >= split_position]

if len(tree_train) < 20 or tree_valid.empty:
    st.warning("There is not enough history for a chronological 100-day Decision Tree evaluation.")
else:
    feature_columns = list(features.columns)
    tree_model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)
    tree_model.fit(tree_train[feature_columns], tree_train["Target"])
    tree_predictions = tree_model.predict(tree_valid[feature_columns])
    tree_actual = tree_valid["Target"].to_numpy()
    tree_rmse = float(np.sqrt(mean_squared_error(tree_actual, tree_predictions)))
    forecast_model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)
    forecast_model.fit(tree_data[feature_columns], tree_data["Target"])
    future_estimate = float(forecast_model.predict(features.tail(1)[feature_columns])[0])

    tree_metrics = st.columns(2)
    tree_metrics[0].metric("100-day holdout RMSE", f"${tree_rmse:,.2f}")
    tree_metrics[1].metric("Estimated close in 100 trading days", f"${future_estimate:,.2f}")
    target_dates = df.index[tree_valid["Target position"].astype(int)]
    tree_chart = pd.DataFrame(
        {"Actual close": tree_actual, "Tree estimate": tree_predictions},
        index=target_dates,
    )
    st.line_chart(tree_chart, height=420)

st.caption("Forecasts are estimates based on historical prices and are not investment advice.")
