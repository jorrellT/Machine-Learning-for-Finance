
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import optuna
import mlflow
import joblib
import os

class MarketModelPipeline:
    def __init__(self, use_hmm=False, n_regimes=3, hmm_features=None, label_horizon='1min', model_dir='saved_models'):
        self.use_hmm = use_hmm
        self.n_regimes = n_regimes
        self.hmm = None
        self.models = {}
        self.scaler = None
        self.hmm_features = hmm_features if hmm_features else ['log_return_z']
        self.final_features = []
        self.label_horizon = label_horizon
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def engineer_features(self, df, rolling_window=18):
        df = df.copy()
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        df['log_return_z'] = (df['log_return'] - df['log_return'].rolling(rolling_window).mean()) / df['log_return'].rolling(rolling_window).std()
        df['buy_sell_pressure_z'] = (df['buy_sell_pressure'] - df['buy_sell_pressure'].rolling(rolling_window).mean()) / df['buy_sell_pressure'].rolling(rolling_window).std()
        df['volume_z'] = (df['volume'] - df['volume'].rolling(rolling_window).mean()) / df['volume'].rolling(rolling_window).std()
        df['vwap_z'] = (df['vwap'] - df['vwap'].rolling(rolling_window).mean()) / df['vwap'].rolling(rolling_window).std()
        return df.dropna()

    def label_data(self, df):
        if self.label_horizon == '1min':
            df['target'] = (df['price'].shift(-6) > df['price']).astype(int)
        elif self.label_horizon == '10s':
            df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
        else:
            raise ValueError("label_horizon must be '10s' or '1min'")
        return df.dropna()

    def train_hmm(self, df):
        X = df[self.hmm_features]
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.hmm = GaussianHMM(n_components=self.n_regimes, covariance_type='full', n_iter=200)
        self.hmm.fit(X_scaled)
        joblib.dump(self.hmm, os.path.join(self.model_dir, 'hmm_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        df['regime'] = self.hmm.predict(X_scaled)
        return df

    def train_model_per_regime(self, df, features, study_trials=25):
        self.final_features = features
        df = df.dropna(subset=['target'])
        mlflow.start_run()
        for regime in sorted(df['regime'].unique()):
            regime_df = df[df['regime'] == regime]
            X = regime_df[features]
            y = regime_df['target']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'tree_method': 'hist',
                    'eval_metric': 'logloss'
                }
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
                preds = model.predict_proba(X_val)[:, 1]
                return log_loss(y_val, preds)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=study_trials)
            best_params = study.best_params
            best_params['tree_method'] = 'hist'
            best_params['eval_metric'] = 'logloss'

            model = xgb.XGBClassifier(**best_params)
            model.fit(X_train, y_train)
            self.models[regime] = model
            joblib.dump(model, os.path.join(self.model_dir, f'model_regime_{regime}.pkl'))

            mlflow.log_params({f"{regime}_{k}": v for k, v in best_params.items()})

        mlflow.end_run()

    def train_model_single(self, df, features, study_trials=25):
        self.final_features = features
        df = df.dropna(subset=['target'])
        X = df[features]
        y = df['target']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        mlflow.start_run()

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'tree_method': 'hist',
                'eval_metric': 'logloss'
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            return log_loss(y_val, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=study_trials)
        best_params = study.best_params
        best_params['tree_method'] = 'hist'
        best_params['eval_metric'] = 'logloss'

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        self.models['all'] = model
        joblib.dump(model, os.path.join(self.model_dir, 'xgb_model.pkl'))

        mlflow.log_params(best_params)
        mlflow.end_run()

    def predict(self, df):
        df = df.copy()
        df = df.dropna(subset=self.final_features)
        if self.use_hmm:
            X_scaled = self.scaler.transform(df[self.hmm_features])
            df['regime'] = self.hmm.predict(X_scaled)
            df['pred_prob'] = df.apply(lambda row: self.models.get(row['regime'], self.models[0]).predict_proba([row[self.final_features]])[0][1], axis=1)
        else:
            df['pred_prob'] = self.models['all'].predict_proba(df[self.final_features])[:, 1]
        df['pred_label'] = (df['pred_prob'] > 0.5).astype(int)
        return df
