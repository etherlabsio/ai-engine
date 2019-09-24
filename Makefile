AWS_ACCESS_KEY_ID=$(shell aws configure get aws_access_key_id --profile ${AWS_PROFILE})
AWS_SECRET_ACCESS_KEY=$(shell aws configure get aws_secret_access_key --profile ${AWS_PROFILE})
AWS_REGION=$(shell aws configure get region --profile ${AWS_PROFILE})
LOGIN=$(shell AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} aws ecr get-login --no-include-email --region ${AWS_REGION})


CONTAINER_IMAGE=817390009919.dkr.ecr.us-east-1.amazonaws.com/etherlabs/keyphrase
STAGING2_IMAGE=933389821341.dkr.ecr.us-east-1.amazonaws.com/etherlabs/keyphrase

ACTIVE_ENV=staging2
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"
BRANCH=$(shell git rev-parse --short HEAD || echo -e '$CI_COMMIT_SHA')
ARTIFACT=keyphrase-server
SERVICE_NAME=keyphrase-service


pre-deploy-notify:
	@curl -X POST --data-urlencode 'payload={"text": "[${ENVIRONMENT}] [${BRANCH}] ${USER}: ${ARTIFACT} is being deployed"}' \
				 ${SLACK_WEBHOOK_URL}

post-deploy-notify:
	@curl -X POST --data-urlencode 'payload={"text": "[${ENVIRONMENT}] [${BRANCH}] ${USER}: ${ARTIFACT} is deployed"}' \
				 ${SLACK_WEBHOOK_URL}

build: clean
	echo ${ACTIVE_ENV} ${ARTIFACT} ${CONTAINER_IMAGE} ${BRANCH}
	eval ${LOGIN}
	@docker build --build-arg active_env=${ACTIVE_ENV} --build-arg app=keyphrase -t ${CONTAINER_IMAGE}:${CONTAINER_TAG} .
	@docker push ${CONTAINER_IMAGE}:${CONTAINER_TAG}
	@docker tag ${CONTAINER_IMAGE}:${CONTAINER_TAG} ${CONTAINER_IMAGE}:${BRANCH}
	@docker push ${CONTAINER_IMAGE}:${BRANCH}

deploy_ecs: build
	$(MAKE) pre-deploy-notify
	ecs deploy ${CLUSTER_NAME} ${SERVICE_NAME} --timeout 600 --profile ${AWS_PROFILE} --task keyphrase --tag ${BRANCH}
	$(MAKE) post-deploy-notify

deploy-staging2:
	$(MAKE) deploy_ecs ARTIFACT=${ARTIFACT} CONTAINER_TAG=staging2 CONTAINER_IMAGE=${STAGING2_IMAGE} \
			ENVIRONMENT=staging2 CLUSTER_NAME=ml-inference SERVICE_NAME=${SERVICE_NAME} AWS_PROFILE=staging2

deploy-production:
	$(MAKE) deploy_ecs ARTIFACT=${ARTIFACT} CONTAINER_TAG=latest CONTAINER_IMAGE=${CONTAINER_IMAGE} \
			ENVIRONMENT=production CLUSTER_NAME=ml-inference SERVICE_NAME=${SERVICE_NAME} AWS_PROFILE=default

.PHONY: dependencies
dependencies:
	pipenv install

.PHONY: version
version:
	git rev-parse HEAD > .version


# .PHONY: build
# build: version
#     @eval ${LOGIN}
# 	sudo docker build -t ${CONTAINER_IMAGE}:staging2 . --build-arg app=${app}
# 	docker push ${CONTAINER_IMAGE}:staging2
# 	docker tag ${CONTAINER_IMAGE}:${CONTAINER_TAG} ${CONTAINER_IMAGE}:${BRANCH}
# 	docker push ${CONTAINER_IMAGE}:${BRANCH}

.PHONY: production
production: version
	sudo docker build -t ${CONTAINER_IMAGE} . --build-arg app=${app}
	docker push ${CONTAINER_IMAGE}

.PHONY: clean
clean:
	rm -f .version

.PHONY: run
run:
	sudo ./pants run cmd/${app}-server:server

.PHONY: binary
binary:
	sudo ./pants binary cmd/${app}-server:server

.PHONY: docker-build
docker-build:
	sudo docker build . --tag ${app} --build-arg app=${app}

.PHONY: docker-run
docker-run:
	sudo docker run ${app}

.PHONY: new-service
new-service:
	@mkdir services/${app}
	@mkdir cmd/${app}-server
	@touch cmd/${app}-server/main.py
	@cp .template/BUILD.cmd cmd/${app}-server/BUILD
	@cp .template/BUILD.services services/${app}/BUILD
	@echo -e '\n\n Added cmd/${app}-server with BUILD & main file \n Added services/${app} with BUILD file'
	@echo -e '\nNote: Kindly go into the Build files present in the 'services/${app}/' and 'cmd/${app}-server/'. \n Change the service name from Keyphrase to your respective service name and, \n add/remove the dependencies mentioned in the cmd/${app}-server/BUILD file '

.PHONY: update-vendor
update-vendor:
	@mkdir vendor
	@cp .template/BUILD.vendor vendor/BUILD
	@wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz
	@tar -xvf en_core_web_sm-2.1.0.tar.gz
	@mv en_core_web_sm-2.1.0/en_core_web_sm vendor/
	@rm -rf en_core_web_sm-2.1.0
	@rm -rf en_core_web_sm-2.1.0.tar.gz

.PHONY: update-lambda-function-mind
update-lambda-function-mind:
	aws lambda update-function-code --function-name mind-01daayheky5f4e02qvrjptftxv --s3-bucket io.etherlabs.artifacts --s3-key ${env_bucket}/mind-serving-lambda.pex --profile ${ENV}
	aws lambda update-function-code --function-name mind-01daatanxnrqa35e6004hb7mbn --s3-bucket io.etherlabs.artifacts --s3-key ${env_bucket}/mind-serving-lambda.pex --profile ${ENV}
	aws lambda update-function-code --function-name mind-01daaqy88qzb19jqz5prjfr76y --s3-bucket io.etherlabs.artifacts --s3-key ${env_bucket}/mind-serving-lambda.pex --profile ${ENV}
	aws lambda update-function-code --function-name mind-01daatbc3ak1qwc5nyc5ahv2xz  --s3-bucket io.etherlabs.artifacts --s3-key ${env_bucket}/mind-serving-lambda.pex --profile ${ENV}
	aws lambda update-function-code --function-name mind-01dadp74wfv607knpcb6vvxgtg --s3-bucket io.etherlabs.artifacts --s3-key ${env_bucket}/mind-serving-lambda.pex --profile ${ENV}

.PHONY: uploadtos3-mind
uploadtos3-mind:
	aws s3 cp --profile production dist/mind-serving-lambda.pex s3://io.etherlabs.artifacts/${ENV}/mind-serving-lambda.pex

.PHONY: update-lambda-function-pims
update-lambda-function-pims:
	aws s3 cp --profile production dist/pims-serving-lambda.pex s3://io.etherlabs.artifacts/${ENV}/pims-serving-lambda.pex
	aws lambda update-function-code --function-name pim --s3-bucket io.etherlabs.artifacts --s3-key ${ENV}/pims-serving-lambda.pex
