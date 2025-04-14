
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score

def evaluate_predictions(
    df, model_dict, hmm_model, features, lookback=30, pip_size=0.0001, min_pip_move=1,
    window_type='sliding', store_results=True, visualize=True
):
    df = df.copy()
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df['log_return_z'] = (df['log_return'] - df['log_return'].rolling(lookback).mean()) / df['log_return'].rolling(lookback).std()
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    df.dropna(inplace=True)

    pip_threshold = pip_size * min_pip_move
    pred_data = []

    for i in range(lookback * 2, len(df)):
        window_df = df.iloc[i - lookback:i] if window_type == 'sliding' else df.iloc[:i]
        current_row = df.iloc[[i]]

        # 30-minute pip move condition (start vs. now)
        price_now = current_row['price'].values[0]
        price_then = window_df.iloc[0]['price']
        if abs(price_now - price_then) < pip_threshold:
            continue  # Skip if no meaningful move

        # Predict regime using full rolling window context
        context_window = df['log_return_z'].iloc[i - lookback:i].values.reshape(-1, 1)
        regime = hmm_model.predict(context_window)[-1]
        model = model_dict.get(regime)

        if model:
            X_live = current_row[features]
            dmatrix = xgb.DMatrix(X_live)
            pred = model.predict(dmatrix)[0]
        else:
            pred = 0.5

        pred_data.append({
            'timestamp': current_row['timestamp'].values[0],
            'price': price_now,
            'price_change': price_now - price_then,
            'true': current_row['target'].values[0],
            'pred_prob': pred,
            'pred_label': int(pred > 0.5),
            'regime': regime
        })

    results_df = pd.DataFrame(pred_data)
    metrics = {
        'accuracy': accuracy_score(results_df['true'], results_df['pred_label']),
        'log_loss': log_loss(results_df['true'], results_df['pred_prob']),
        'precision': precision_score(results_df['true'], results_df['pred_label']),
        'recall': recall_score(results_df['true'], results_df['pred_label'])
    }

    if store_results:
        results_df.to_csv("live_predictions.csv", index=False)

    if visualize:
        import matplotlib.pyplot as plt
        results_df.set_index('timestamp', inplace=True)
        results_df['rolling_acc'] = results_df['true'].eq(results_df['pred_label']).rolling(50).mean()
        results_df['pred_prob'].rolling(50).mean().plot(label='Smoothed Prob', alpha=0.7)
        results_df['rolling_acc'].plot(secondary_y=True, label='Rolling Accuracy', linestyle='--', alpha=0.7)
        plt.title("Prediction Probabilities and Rolling Accuracy (HMM contextual)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return metrics, results_df
