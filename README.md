# ☁️ MLOps Cloud Project - Weather Forecast Pipeline

## 🚀 CI/CD Codecov 커버리지

[![codecov](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO_NAME/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO_NAME)
[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/build-deploy.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/build-deploy.yml)


<br>

## 💻 프로젝트 소개 & 🔨 개발 환경 및 기술 스택
This project implements a full **MLOps pipeline** for time-series **weather forecasting** and **clothing recommendation** service, based on:
- **Airflow** for orchestration
- **MLflow** for experiment tracking & model management
- **Project_root** pipeline for training & inference
- **Serving layer** for API and visualization
- **Streamlit** for internal monitoring UI
- **Grafana** for metric dashboards 
- **Prometheus** for metrics collection & storage & alerting

<br>

## 👨‍👩‍👦‍👦 팀 구성원

| <img src="https://github.com/ohseungtae.png" width="120"/> | <img src="https://github.com/JBreals.png" width="120"/> | <img src="https://github.com/kdlee02.png" width="120"/> | <img src="https://github.com/hwang1999.png" width="120"/> | 
| :--------------------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: | :--------------------------------------------------------: | 
| [오승태](https://github.com/ohseungtae)                   | [김정빈](https://github.com/JBreals)                   | [이경도](https://github.com/kdlee02)                   | [황준엽](https://github.com/hwang1999)                   |
| ML 모델 구축 & CI CD 담당                                   |     Docke repo 구축 & <br>배포 파이프라인 자동화 & <br>인프라 구축                         |      ML Engineering & <br>Monitoring & Streamlit                           |            ML 모델 구축 & <br>추론 API 코드 설계                     |     


<br>

## 📁 프로젝트 구조
```
├── api_server-network/
│   ├── api-server/              # API Gateway (FastAPI)
│   ├── inference-server/        # Inference Backend (FastAPI)
│   ├── streamlit/               # Visualization UI (Streamlit)
├── datapipeline/                # Data pipeline scripts
├── docs/                        # Project documentation
├── mlops-airflow/               # Airflow DAGS & Plugins
├── mlops-mlflow/                # MLflow server + example experiments
├── modeling/                    # Prototype notebooks
├── project_root/                # Production pipeline code (src/main.py → run_all)
├── serving/                     # Serving utils
├── monitoring/                  # Prometheus and Grafana
├── docker-compose.yml           # Project orchestration (Docker Compose)
├── requirements.txt             # Core dependencies
└── README.md                    # Project documentation

```

<br>

## 💻​ 구현 기능
### 기능1
- **모델 학습 자동화 파이프라인 구축**
모델 학습(train), 검증(eval), 추론(inference)까지의 전체 워크플로우를 자동화하여 CLI 명령어 또는 API 요청으로 학습/추론 가능
### 기능2
- **REST API 기반 추론 서비스 제공**
학습된 모델을 기반으로 API 서버를 통해 외부 서비스나 클라이언트가 실시간으로 예측 결과 요청 및 수신 가능
### 기능3
- **모델 버전 관리 및 아티팩트 저장소 연동**
학습된 모델 아티팩트를 자동 저장 및 버전 관리하며, MLflow 연동으로 실험 결과 추적 가능
### 기능4
- **네트워크 기반 분산 환경 대응 구조 설계**
네트워크 계층과 API 서버를 분리하여 확장성과 유지보수성을 고려한 MSA 구조 적용
### 기능5
- **데이터 파이프라인 구성 및 전처리 자동화**
raw 데이터를 수집하여 가공/전처리 후 모델 학습에 최적화된 데이터셋 생성까지 자동으로 처리

<br>

## 🛠️ 작품 아키텍처(필수X)
## 🗺️ Architecture

```plaintext
┌───────────────────────────────────────────────────────┐
│                       Airflow                         │
│ weather_pipeline_dag.py → Data Ingestion              │
│ model_deployment_dag.py→inference-server/run_inference│
└───────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────┐
│                    Data Pipeline                      │
│ collect_tokyo_weather → load_and_split                │
└───────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────┐
│                       Modeling                        │
│ train_prophet / train_sarimax → MLflow Tracking       │
└───────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────┐
│                Evaluation & Testing                   │
│ evaluate_* → predict_testset                          │
└───────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────┐
│                    Model Selection                    │
│ get_best_model                                        │
└───────────────────────────────────────────────────────┘
                           ↓                           
┌───────────────────────────────────────────────────────┐
│                       CI & CD                         │
│ unit test -> integration-test -> build docker         │
└───────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────┐
│                    Deployment (Airflow)               │
│ model_deployment.py → inference-server/run_inference  │
└───────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────┐
│                    Serving Layer                      │
│ api-server: /forecast, /clothing                      │
│ inference-server: /run_inference                      │
│ streamlit: Visualization Dashboard                    │
└───────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────┐
│                      Monitoring                       │
│ Prometheus                                            │
│ Grafana                                               │
└───────────────────────────────────────────────────────┘
```
<br>

## 📝 Future Improvements

- Add model monitoring and alerting
- Integrate CI/CD pipeline
- Automate Airflow DAG triggering
- Improve API authentication
- Utilize rabbit mq for alerting messages + slack/email
- collect metric data for each containers and services using prometheus

