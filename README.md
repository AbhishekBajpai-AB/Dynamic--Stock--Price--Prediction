## Stock Price Prediction

A Streamlit dashboard for viewing Yahoo Finance history and comparing three separate forecasting approaches: K-Nearest Neighbors (KNN), Long Short-Term Memory (LSTM), and Decision Tree regression.

## 🌐 Live Demo

Visit the live platform: [Stock Price Prediction](https://dynamic--stock--price--prediction--abhishek.streamlit.app/)

## What the app shows

- Daily open, high, low, close, and volume data for the selected ticker and date range.
- KNN next-day direction classification and next-day closing-price estimates, evaluated on the latest chronological holdout period.
- An LSTM trained for the selected ticker, with its final 20% of dates held out for evaluation.
- A Decision Tree estimate 100 trading days ahead, evaluated on examples whose target dates fall in the latest 20% of the selected history.

The models are evaluated separately; their outputs are not combined into an ensemble. Forecasts use historical market data and are estimates, not investment advice.

## Run locally on Windows

Use 64-bit Python 3.11. From PowerShell in the repository folder:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run "MERGED MODELS/real.py"
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. The first LSTM run can take a little while; results are cached for the same stock and history. An internet connection is required to download market prices from Yahoo Finance.

## Deploy on Streamlit Community Cloud

Choose the GitHub repository `AbhishekBajpai-AB/Dynamic--Stock--Price--Prediction`, branch `main`, and entrypoint `MERGED MODELS/real.py`. Select Python 3.11 in the app's advanced settings. The root `requirements.txt` installs the app dependencies.

## Project files

- `MERGED MODELS/real.py` — Streamlit dashboard and prediction workflow.
- `MERGED MODELS/lstm_trainer.py` — isolated TensorFlow training process for the LSTM forecast.
- `LSTM/` — LSTM training notebooks and the notebook's saved model artifact.
- `KNN/` — KNN exploration notebook.
- `Decision Tree/` — Decision Tree notebook and its training artifact.
- `PROJECT REPORT/` and `VISUALISATION/` — project report and figures.

## Data and limitations

Yahoo Finance availability, symbols, and historical coverage can change. The app stops with a message when no data is returned or when the selected range is too short. Model scores describe the selected historical holdout only; they do not guarantee future performance.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
