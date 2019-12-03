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


.PHONY: new-service
new-service:
	@mkdir services/${app}
	@mkdir cmd/${app}-server
	@touch cmd/${app}-server/main.py
	@cp .template/BUILD.cmd cmd/${app}-server/BUILD
	@cp .template/BUILD.services services/${app}/BUILD
	@echo -e '\n\n Added cmd/${app}-server with BUILD & main file \n Added services/${app} with BUILD file'
	@echo -e '\nNote: Kindly go into the Build files present in the 'services/${app}/' and 'cmd/${app}-server/'. \n Change the service name from Keyphrase to your respective service name and, \n add/remove the dependencies mentioned in the cmd/${app}-server/BUILD file '


.PHONY: clean
clean:
	rm -f .version

.PHONY: dependencies.pex

.PHONY: update-lambda-function-scorer
update-lambda-function-scorer:
	aws s3 cp --profile production dist/scorer_lambda.pex s3://io.etherlabs.artifacts/${ENV}/scorer_lambda.pex
	aws lambda update-function-code --function-name pim --s3-bucket io.etherlabs.artifacts --s3-key ${ENV}/scorer_lambda.pex

.PHONY: update-lambda-function-gs
update-lambda-function-gs:
	./pants bundle cmd/group_segments-server:group_segments_code
	aws s3 cp --profile ${ENV} dist/group_segments_code.pex s3://io.etherlabs.artifacts/${ENV}/group_segments_code.pex
	aws lambda update-function-code --function-name group-segments --s3-bucket io.etherlabs.artifacts --s3-key ${ENV}/group_segments_code.pex

.PHONY: test-gs
test-gs:
	aws s3 cp --profile staging2 dist/group_segments_code.pex s3://io.etherlabs.staging2.contexts/topics/group_segments_code.pex
	aws lambda update-function-code --function-name pex_test --s3-bucket io.etherlabs.staging2.contexts --s3-key topics/group_segments_code.pex --region=us-east-1

.PHONY: update-lambda-function-sa
update-lambda-function-sa:
	./pants bundle cmd/segment_analyzer-server:segment_analyser_lambda
	aws s3 cp --profile ${ENV} dist/segment_analyser_lambda.pex s3://io.etherlabs.artifacts/${ENV}/segment_analyser_lambda.pex
	aws lambda update-function-code --function-name segment-analyser --s3-bucket io.etherlabs.artifacts --s3-key ${ENV}/segment_analyser_lambda.pex  --profile ${ENV} 

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

.PHONY: build-upload-update-lambda
build-upload-update-lambda:
	./pants bundle cmd/${app-name}-server:${build-name}
	aws s3 cp --profile production dist/${build-name}.pex s3://io.etherlabs.artifacts/${ENV}/${build-name}.pex
	aws lambda update-function-code --function-name ${function-name} --s3-bucket io.etherlabs.artifacts --s3-key ${ENV}/${build-name}.pex

