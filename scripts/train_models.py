import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'src' / 'models'
MODEL_OUTPUT = MODEL_DIR / 'trained'
METRICS_OUTPUT = MODEL_DIR / 'training_metrics.json'
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

DATA_FILE = ROOT / 'data' / 'clean' / 'olist_ml_ready.csv'
df = pd.read_csv(DATA_FILE)
print(f'Dataset completo carregado com {len(df)} registros.')

drop_cols = [
    'order_id', 'customer_id',
    'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date'
]
TARGET = 'delivery_time_days'
X = df.drop(columns=drop_cols + [TARGET], errors='ignore')
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    'RandomForest': (
        RandomForestRegressor(random_state=42),
        {
            'n_estimators': [50, 70],
            'max_depth': [10, 15],
            'min_samples_leaf': [1, 2]
        }
    ),
    'GradientBoosting': (
        GradientBoostingRegressor(random_state=42),
        {
            'n_estimators': [40, 60],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4]
        }
    ),
    'HistGradientBoosting': (
        HistGradientBoostingRegressor(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ),
        {
            'max_iter': [80, 100],
            'learning_rate': [0.05, 0.08],
            'max_depth': [10, 12]
        }
    )
}

searches = {}
for name, (estimator, params) in models.items():
    print(f'Executando {name}...')
    search = RandomizedSearchCV(
        estimator,
        params,
        n_iter=1,
        cv=2,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=1
    )
    search.fit(X_train, y_train)
    searches[name] = search
    print(f'  Melhor RMSE CV: {-search.best_score_:.3f}')

metrics = []
for name, search in searches.items():
    preds = search.best_estimator_.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    metrics.append({
        'modelo': name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2_score(y_test, preds)
    })
    joblib.dump(search.best_estimator_, MODEL_OUTPUT / f'{name}.joblib')

metrics_df = pd.DataFrame(metrics).sort_values('rmse')
metrics_df.to_json(METRICS_OUTPUT, orient='records', force_ascii=False, indent=2)
print(f'Treinamento concluído. Métricas gravadas em {METRICS_OUTPUT}')
