
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, log_loss, precision_score, recall_score
import numpy as np

def evaluate_predictions(df, label_col='target', pred_col='pred_label', prob_col='pred_prob'):
    print("=== Classification Report ===")
    print(classification_report(df[label_col], df[pred_col]))

    metrics = {
        'accuracy': accuracy_score(df[label_col], df[pred_col]),
        'log_loss': log_loss(df[label_col], df[prob_col]),
        'precision': precision_score(df[label_col], df[pred_col]),
        'recall': recall_score(df[label_col], df[pred_col])
    }
    return metrics

def backtest_signals(df, label_col='target', pred_col='pred_label', price_col='price', cost_per_trade=0.0):
    df = df.copy()
    df['return'] = df[price_col].shift(-1) / df[price_col] - 1
    df['strategy_return'] = df['return'] * df[pred_col]
    df['pnl'] = df['strategy_return'] - cost_per_trade * (df[pred_col] != df[pred_col].shift(1)).astype(int)

    result = {
        'total_pnl': df['pnl'].sum(),
        'mean_return': df['strategy_return'].mean(),
        'win_rate': (df[pred_col] == df[label_col]).mean(),
        'n_signals': df[pred_col].sum(),
        'sharpe': df['pnl'].mean() / df['pnl'].std() * np.sqrt(252 * 6) if df['pnl'].std() != 0 else 0
    }
    return df, result

def real_time_predict(pipeline, df, lookback=20):
    """ Simulate real-time prediction by rolling over the dataset and predicting step-by-step. """
    results = []
    for i in range(lookback, len(df)):
        batch = df.iloc[i-lookback:i].copy()
        batch_feat = pipeline.engineer_features(batch)
        batch_feat = pipeline.label_data(batch_feat)
        pred_row = pipeline.predict(batch_feat.tail(1))
        results.append(pred_row)
    return pd.concat(results).reset_index(drop=True)
