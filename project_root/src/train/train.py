import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from tqdm import tqdm
from icecream import ic
from src.utils.utils import model_dir, ensure_dir
import mlflow

def train_prophet(
    train_csv,
    model_name='prophet_model.pkl',
    seasonality_mode='multiplicative',
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=15.0,
    **kwargs
):
    ensure_dir(model_dir())
    df = pd.read_csv(train_csv, parse_dates=['time'])
    data = df[['time', 'temp']].copy()
    data.columns = ['ds', 'y']
    with mlflow.start_run():

        mlflow.log_param("seasonality_mode", seasonality_mode)
        mlflow.log_param("daily_seasonality", daily_seasonality)
        mlflow.log_param("weekly_seasonality", weekly_seasonality)
        mlflow.log_param("yearly_seasonality", yearly_seasonality)
        mlflow.log_param("changepoint_prior_scale", changepoint_prior_scale)
        mlflow.log_param("seasonality_prior_scale", seasonality_prior_scale)

        model = Prophet(
        seasonality_mode=seasonality_mode,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        **kwargs
    )
        ic("Fitting Prophet model...")
        model.fit(data)
        model_path = model_dir(model_name)
        joblib.dump(model, model_path)
        ic(f"Model saved to {model_path}")
        return model_path, run.info.run_id

def train_sarimax(
    train_csv,
    test_csv=None,
    model_name="sarimax_model.pkl",
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False,
    use_gridsearch=False,
    param_grid=None
):
    ensure_dir(model_dir())
    df = pd.read_csv(train_csv, parse_dates=["time"])
    df = df.sort_values("time")
    y = df["temp"].values

    if not use_gridsearch:
        ic(f"Training SARIMAX model with order={order} and seasonal_order={seasonal_order}")
        with mlflow.start_run():
            mlflow.log_param("order", order)
            mlflow.log_param("seasonal_order", seasonal_order)
            mlflow.log_param("enforce_stationarity", enforce_stationarity)
            mlflow.log_param("enforce_invertibility", enforce_invertibility)

            model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            )

            results = model.fit(disp=False)
            model_path = model_dir(model_name)
            joblib.dump(results, model_path)
            ic(f"SARIMAX model saved to {model_path}")
            return model_path, run.info.run_id
    else:
        assert test_csv is not None, "GridSearch 시 test_csv가 필요합니다."

        if param_grid is None:
            param_grid = {
            "p": [0, 1],
            "d": [0, 1],
            "q": [0, 1],
            "P": [0, 1],
            "D": [0, 1],
            "Q": [0, 1],
            "s": 24
        }
        df_test = pd.read_csv(test_csv, parse_dates=["time"])
        y_test = df_test["temp"].values

        # 탐색 범위 설정
        p_values = param_grid.get("p", [0, 1])
        d_values = param_grid.get("d", [0, 1])
        q_values = param_grid.get("q", [0, 1])
        P_values = param_grid.get("P", [0, 1])
        D_values = param_grid.get("D", [0, 1])
        Q_values = param_grid.get("Q", [0, 1])
        s = param_grid.get("s", 24)

        best_score = float("inf")
        best_order = None
        best_seasonal_order = None
        best_model = None

        ic("🔍 Grid Search 시작...")
        for order_combo in tqdm(itertools.product(p_values, d_values, q_values)):
            for seasonal_combo in itertools.product(P_values, D_values, Q_values):
                seasonal_order_combo = seasonal_combo + (s,)
                try:
                    model = SARIMAX(
                        y,
                        order=order_combo,
                        seasonal_order=seasonal_order_combo,
                        enforce_stationarity=enforce_stationarity,
                        enforce_invertibility=enforce_invertibility
                    )
                    result = model.fit(disp=False)
                    y_pred = result.predict(start=0, end=len(y_test)-1)
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                    ic(order_combo, seasonal_order_combo, rmse)

                    if rmse < best_score:
                        best_score = rmse
                        best_order = order_combo
                        best_seasonal_order = seasonal_order_combo
                        best_model = result

                except Exception as e:
                    ic(f"❌ 실패: order={order_combo}, seasonal={seasonal_order_combo}, 에러: {e}")
                    continue
        
        with mlflow.start_run():
            mlflow.log_param("order", best_order)
            mlflow.log_param("seasonal_order", best_seasonal_order)
            mlflow.log_param("enforce_stationarity", enforce_stationarity)
            mlflow.log_param("enforce_invertibility", enforce_invertibility)
        # 최종 모델 저장
        if best_model:
            model_path = model_dir(model_name)
            joblib.dump(best_model, model_path)
            ic(f"✅ 최적 모델 저장 완료: order={best_order}, seasonal_order={best_seasonal_order}, RMSE={best_score}")
            return model_path, run.info.run_id
        else:
            raise ValueError("모든 조합에서 모델 학습 실패")