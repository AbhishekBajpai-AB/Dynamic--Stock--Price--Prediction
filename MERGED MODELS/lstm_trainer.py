"""Train one stock-specific LSTM in an isolated process."""

import sys

import numpy as np
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed


LOOKBACK_DAYS = 60


def main(input_path: str, output_path: str) -> None:
    with np.load(input_path, allow_pickle=False) as history:
        values = np.asarray(history["close"], dtype=np.float32).reshape(-1, 1)
        dates = history["dates"].astype(str)

    train_end = int(len(values) * 0.8)
    if train_end <= LOOKBACK_DAYS or len(values) <= train_end:
        raise ValueError("There is not enough history to train and validate the LSTM.")

    train_min = float(np.min(values[:train_end]))
    train_max = float(np.max(values[:train_end]))
    scale = 1.0 / (train_max - train_min) if train_max > train_min else 1.0
    train_scaled = (values[:train_end] - train_min) * scale
    x_train = np.asarray(
        [train_scaled[index - LOOKBACK_DAYS:index] for index in range(LOOKBACK_DAYS, train_end)],
        dtype=np.float32,
    )
    y_train = train_scaled[LOOKBACK_DAYS:, 0]

    test_values = values[train_end - LOOKBACK_DAYS:]
    test_scaled = (test_values - train_min) * scale
    x_test = np.asarray(
        [test_scaled[index - LOOKBACK_DAYS:index] for index in range(LOOKBACK_DAYS, len(test_scaled))],
        dtype=np.float32,
    )
    actual = values[train_end:, 0]
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("There is not enough history to build LSTM sequences.")

    set_random_seed(42)
    model = Sequential(
        [
            LSTM(32, input_shape=(LOOKBACK_DAYS, 1)),
            Dropout(0.1),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        x_train,
        y_train,
        epochs=12,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=0,
    )
    predicted_scaled = model.predict(x_test, verbose=0)
    predicted = predicted_scaled.ravel() / scale + train_min
    keras_backend.clear_session()

    np.savez_compressed(
        output_path,
        actual=actual,
        predicted=predicted,
        dates=dates[train_end:],
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: lstm_trainer.py INPUT.npz OUTPUT.npz")
    main(sys.argv[1], sys.argv[2])
