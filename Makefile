LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME_API:=escooters-trips/escooters-trips-api:${LOCAL_TAG}
LOCAL_IMAGE_NAME_TRAIN:=escooters-trips/escooters-trips-train:${LOCAL_TAG}

test:
	pytest tests/

quality_checks: test
	isort .
	black .
	pylint --recursive=y --fail-under=9 .

build_api: quality_checks
	docker build --platform linux/amd64 -f .docker/Dockerfile-api -t ${LOCAL_IMAGE_NAME_API} .

build_train: quality_checks
	docker build --platform linux/amd64 -f .docker/Dockerfile-train -t ${LOCAL_IMAGE_NAME_TRAIN} .
