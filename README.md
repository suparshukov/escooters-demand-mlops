# Project: Forecasting the demand for electric scooters in Chicago neighbourhoods

## 1. Problem description


The company rents electric scooters in 77 Chicago neighbourhoods. Each neighbourhood has a small logistics centre - a warehouse where electric scooters are concentrated, from which the employees responsible for the neighbourhood distribute the scooters throughout the day to different locations within the neighbourhood. In order to ensure the required number of scooters in the district warehouse, it is necessary to have a forecast of the demand for scooters for the next 3 days. Such forecasting horizon is conditioned by the process of logistic supply of electric scooters to the district warehouse from remote warehouses outside the city, electric scooter service stations and the process of purchasing new electric scooters from suppliers. The task of distributing electric scooters within the districts during the day has been solved.

The purpose of the Project is to provide the Company with forecasts of demand for electric scooters in Chicago neighbourhoods in order to meet this demand with electric scooters. The use of such forecasts will minimise scooter downtime (not keeping excess scooters in district warehouses) and minimise the occurrence of scooter shortage situations in district warehouses to meet demand within the district. The Project will develop a solution that will generate forecasts of the number of rides on the Company's electric scooters in each of Chicago's 77 neighbourhoods. Forecasting will be done for the next 3 days with a granularity of 1 day for each neighbourhood.


## Project Goals
To implement and test solutions based on ML models to predict the demand for electric scooters in Chicago's inner city neighbourhoods The workability of the solution is validated on a dataset that was not used in the development of the Solution; The quality of the models is evaluated:
1. Expertly;
2. By calculating metrics:
   - MAPE error of forecasting the number of trips within the city,
   - MAPE error in predicting the number of trips for each neighbourhood,
   neighbourhood scale-weighted APE trip forecast error,
   - BCR(N) (bad case rate - proportion of neighbourhoods with MAPE trip forecast error > N%),
   - the ratio of the predicted number of trips to the actual number of trips (for neighbourhoods separately and for the city as a whole).

   The metrics will be calculated based on the forecast for the next 3 days with a granularity of 1 day for each neighbourhood and for the city as a whole.

## Project Tasks
The project requires the Project team to perform the following tasks:
- Organising a repository to store the Project code and artefacts;
- Organisation of quality assurance of the Solution source code;
- Application of the Solution dependency management tools;
- Uploading and processing of historical data on the Company's electric scooter rides;
- Uploading and processing geographic data related to the city of Chicago;
- Analysis of the uploaded and processed data;
- Developing and evaluating model(s) for predicting the demand for the Company's Electric
- Development (setup and application) of tools(s) to automate reproducible scalable research conducted as part of the Project;
- Development of a service based on ML-model(s) providing forecasts of demand for electric scooters by districts, using containerisation (Docker);
- Organisation of automated testing of parts of the Solution;
- Development of monitoring and analytical tools for the service and model(s);
- Organisation of automated deployment of the developed Solution in a productive environment.

## Project deliverables
Project deliverables should include:
Programme code of the Solution
A final report including:
1. Description of the data collection process;
2. Results of data analysis and preprocessing;
3. Metrics of the quality of the model predictions;
4. Comparison of the tested models/approaches and description of the best solution in terms of metrics;
5. Description of the applicability boundary of the Solution;
6. Recommendations for refining the Solution.
7. A link to the Solution deployed on the web (service/application);
8. Presentation with conclusions on the Project outcome, recommendations and conclusion on further development of the Project.


## 2. Cloud

## 3. Experiment tracking and model registry

## 4. Workflow orchestration

## 5. Model deployment

## 6. Model monitoring

## 7. Reproducibility

## 8. Best practices
  - There are unit tests
  - There is an integration test
  - Linter and/or code formatter are used
  - There's a Makefile
  - There are pre-commit hooks
  - There's a CI/CD pipeline
