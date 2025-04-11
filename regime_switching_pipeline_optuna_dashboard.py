
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
import mlflow
import mlflow.xgboost

class RegimeSwitchingPipeline:
    def __init__(self, n_regimes=3, hmm_lookback=30, horizon=1, features=None, use_online=False, tune_hyperparams=False, n_trials=20):
        self.n_regimes = n_regimes
        self.hmm_lookback = hmm_lookback
        self.horizon = horizon
        self.features = features
        self.hmm = None
        self.models = {}
        self.use_online = use_online
        self.tune_hyperparams = tune_hyperparams
        self.n_trials = n_trials

    def generate_labels(self, df):
        df['target'] = (df['price'].shift(-self.horizon) > df['price']).astype(int)
        return df.dropna()

    def detect_regimes(self, df):
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        df['log_return_z'] = (df['log_return'] - df['log_return'].rolling(self.hmm_lookback).mean()) / df['log_return'].rolling(self.hmm_lookback).std()
        df.dropna(inplace=True)
        self.hmm = GaussianHMM(n_components=self.n_regimes, covariance_type="full", n_iter=100)
        self.hmm.fit(df[['log_return_z']])
        df['regime'] = self.hmm.predict(df[['log_return_z']])
        return df

    def objective(self, trial, X, y):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'eta': trial.suggest_float('eta', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        }

        tscv = TimeSeriesSplit(n_splits=3)
        losses = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dval, 'eval')],
                              early_stopping_rounds=10, verbose_eval=False)
            preds = model.predict(dval)
            losses.append(log_loss(y_val, preds))

        return np.mean(losses)

    def train_models(self, df, test_size=0.2):
        df = self.generate_labels(df)
        df = self.detect_regimes(df)
        df = df.dropna()

        results = []

        for regime in range(self.n_regimes):
            data = df[df['regime'] == regime]
            if data.empty: continue

            X = data[self.features]
            y = data['target']

            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            if self.tune_hyperparams:
                study = optuna.create_study(
                    direction='minimize',
                    study_name=f"regime_{regime}_study",
                    storage="sqlite:///optuna_regimes.db",
                    load_if_exists=True
                )
                study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=self.n_trials)
                best_params = study.best_params
                best_params.update({
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'tree_method': 'hist'
                })
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                model = xgb.train(best_params, dtrain, num_boost_round=200)
                self.models[regime] = model
                preds = model.predict(dtest)
            else:
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                best_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 4,
                    'eta': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.7,
                    'tree_method': 'hist'
                }
                model = xgb.train(best_params, dtrain, num_boost_round=200)
                self.models[regime] = model
                preds = model.predict(dtest)

            preds_binary = (preds > 0.5).astype(int)
            metrics = {
                'regime': regime,
                'accuracy': accuracy_score(y_test, preds_binary),
                'log_loss': log_loss(y_test, preds),
                'precision': precision_score(y_test, preds_binary),
                'recall': recall_score(y_test, preds_binary)
            }

            with mlflow.start_run(run_name=f"regime_{regime}"):
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                mlflow.xgboost.log_model(self.models[regime], artifact_path=f"regime_{regime}_model")

            results.append(metrics)

            if self.use_online and not self.tune_hyperparams:
                for i in range(len(X_test)):
                    dnew = xgb.DMatrix(X_test.iloc[i:i+1], label=[y_test.iloc[i]])
                    self.models[regime] = xgb.train(
                        best_params, dnew, num_boost_round=1, xgb_model=self.models[regime]
                    )

        return results
