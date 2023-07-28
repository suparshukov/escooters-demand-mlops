## Project: Forecasting the demand for electric scooters in Chicago neighbourhoods

### 1. Problem description

The company rents electric scooters in 77 Chicago neighbourhoods. Each neighbourhood has a small logistics centre - a warehouse where electric scooters are concentrated, from which the employees responsible for the neighbourhood distribute the scooters throughout the day to different locations within the neighbourhood.   
In order to ensure the required number of scooters in the district warehouse, it is necessary to have a forecast of the demand for scooters for the next 3 days. Such forecasting horizon is conditioned by the process of logistic supply of electric scooters to the district warehouse from remote warehouses outside the city, electric scooter service stations and the process of purchasing new electric scooters from suppliers.   
The purpose of the Project is to provide the Company with forecasts of demand for electric scooters in Chicago neighbourhoods in order to meet this demand with electric scooters. The use of such forecasts will minimise scooter downtime (not keeping excess scooters in district warehouses) and minimise the occurrence of scooter shortage situations in district warehouses to meet demand within the district.   
The Project will develop a solution that will generate forecasts of the number of rides on the Company's electric scooters in each of Chicago's 77 neighbourhoods. Forecasting will be done for the next several days (by day) with a granularity of 1 day for each neighbourhood.

### Project Goal:
Build an end-to-end machine learning project that would provide demand forecasts for electric scooters in the neighbourhoods of Chicago.


### Project Tasks
- Cloud (the project should be developed on the cloud)
- Experiment tracking and model registry should be used
- Workflow orchestration (fully deployed workflow)
- Model deployment (the model deployment code should be containerized and could be deployed to cloud or special tools for model deployment are used)
- Model monitoring that calculates and reports metrics
- Reproducibility (instructions should be clear, it should be easy to run the code, and it should work; the versions for all the dependencies should be specified)

- Best practices should also be used:
  - unit tests
  - integration tests
  - linter and/or code formatters
  - Makefile
  - pre-commit hooks
  - CI/CD pipeline


### Project results

#### 1. Cloud
The project is developed on the cloud.
- **Amazon EC2**: a service based on ML-model, MLflow, Model monitoring(Evidently+Grafana) are deployed on EC2 instance.
- **Amazon S3** -  is used for storing model artifacts (models registry storage), raw data for training, reference data for monitoring etc.
- **Elastic Container Registry (ECR)** - stores Docker images for API (model inference) and training process.
- **Amazon RDS** - is used for experiment tracking by MLflow to store the metadata of experiments.
- **Prefect Cloud**  is used to automate and monitor the managed training workflow

#### 2. Experiment tracking and model registry
MLflow (on Amazon EC2) with artifact storage on Amazon S3 are used for experiment tracking and model registry, runs are stored in PostgreSQL (Amazon RDS).   
MLflow http://16.171.140.74:5000   
Experiments: http://16.171.140.74:5000/#/experiments/1   
Amazon S3 bucket is used to store model artifacts: s3://serjeeon-learning-bucket/escooters-demand/mlflow-artifacts-remote   
As a result of the hyperparameters optimisation, the best model is saved to the model registry: http://16.171.140.74:5000/#/models/escooter-demand-model

#### 3. Workflow orchestration
Prefect is used for Workflow orchestration.
A flow for training the model has been developed, including the optimisation of hyperparameters. The flow consists of several tasks and is scheduled to run using Prefect.

```
prefect cloud login -k {PREFECT_KEY}
prefect deployment build -n escooters-demand-train -p default-agent-pool -q escooters-demand-training src/pipeline/train_pipeline.py:main_flow --cron "0 5 * * *"
prefect deployment apply main_flow-deployment.yaml
prefect agent start --pool default-agent-pool --work-queue escooters-demand-training
```

Prefect deployment configurations: main_flow-deployment.yaml

#### 4. Model deployment
Developed a service based on ML-model, which provides forecasts of demand for electric scooters by districts. 
Containerisation (Docker) is used. Docker image is created and pushed to the ECR public repository when doing CI/CD (public.ecr.aws/h6l8h0t3/escooters-trips/escooters-trips-api:latest).   
The service is available at http://16.171.140.74:9696/predict

Method call example:
```
import requests

input_data = {"community": "LAKE VIEW", "date": "2020-09-11"}
#input_data = {"community": "ENGLEWOOD", "date": "2020-10-23"}
# input_data = {"community": "OHARE", "date": "2020-09-20"}

r = requests.post('http://16.171.140.74:9696/predict', json=input_data)
r.json()
```
Response message example:
```
{'model_version': '71e8d2efac7a4433991aba1a699108e3', 'trips': 1113}
```

#### 5. Model monitoring
Evidently is used to calculate model monitoring metrics.   
Grafana is used to visualise the metrics (http://16.171.140.74:3000).   
The metrics are stored in PostreSQL database.   

A dashboard was developed to monitor the model:   
Home -> Dashboards -> Escooters demand monitoring   
http://16.171.140.74:3000/d/dbba5bf2-9fc8-45e8-9980-121956ec0f4c/escooters-demand-monitoring?orgId=1   
Login ‘admin’, password ‘admin’


#### 6. Reproducibility
To ensure reproducibility dependency management tool Poetry is used.   
The required dependencies are organised by using groups in configuration of the project (pyproject.toml): base, train, dev, api, monitoring.

To run a web service use commands:
```
docker build -t escooters-trips-api  \
	--build-arg AWS_ACCESS_KEY_ID="*****" \
	--build-arg AWS_SECRET_ACCESS_KEY="*****" \
	--build-arg AWS_REGION="eu-north-1" \
	--build-arg AWS_OUTPUT="json" \
	-f .docker/Dockerfile-api .
docker run -p 9696:9696 --rm --name=escooters-trips-api escooters-trips-api
```
or
```
docker pull public.ecr.aws/h6l8h0t3/escooters-trips/escooters-trips-api:latest
docker run -v /home/ec2-user/.aws:/root/.aws -p 9696:9696 --rm public.ecr.aws/h6l8h0t3/escooters-trips/escooters-trips-api:latest
```

To run Model Training use commands:
```
docker build -f .docker/Dockerfile-train -t escooters-trips-train .
docker run --env TRACKING_URI="http://16.171.140.74:5000" --rm --name=escooters-trips-train escooters-trips-train
```
or
```
docker pull public.ecr.aws/h6l8h0t3/escooters-trips/escooters-trips-train:latest
docker run --env TRACKING_URI="http://16.171.140.74:5000" --rm public.ecr.aws/h6l8h0t3/escooters-trips/escooters-trips-train:latest
```

#### 7. Best practices
 - There are unit tests
 - Linter and code formatter are used: pylint, black, isort
 - There's a Makefile with tasks: test (pytest), quality_checks (isort, black, pylint), build_api, build_train
 - There are pre-commit hooks:
   - trailing-whitespace, end-of-file-fixer,check-yaml, check-added-large-files
   - isort
   - black
   - pylint
 - There's a CI/CD pipeline with stages:
   - check_code
   - build-api (build, tag and push an image with the web service to Amazon ECR)
   - build-train (build, tag and push an image with Model Training to Amazon ECR)
