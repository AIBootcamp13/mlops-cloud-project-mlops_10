import mlflow
from mlflow.tracking import MlflowClient
import fire
import os

def register_best_model(metric: str = "mae"):
    # 1. MLflow 환경 설정
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    # 2. 실험 가져오기
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # 3. metric 기준 최적 run 선택
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1
    )
    if not runs:
        raise ValueError("No runs found for best model selection.")

    
    best_run = runs[0]
    model_type = best_run.data.params["model_type"]
    model_name = f"{model_type}_{os.getenv('MLFLOW_EXPERIMENT_NAME')}"
    model_uri = f"runs:/{best_run.info.run_id}/model"
    print(f"📌 Best run_id: {best_run.info.run_id}")
    print(f"📁 Model URI: {model_uri}")

    # 4. 모델 등록
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"✅ Registered model version: {result.version}")

    # 5. 가장 최신 버전을 Production 스테이지로 promote
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"🚀 Model {model_name} v{result.version} promoted to Production")

if __name__ == "__main__":
    fire.Fire(register_best_model)
