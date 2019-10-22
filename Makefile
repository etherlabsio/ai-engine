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

.PHONY: update-lambda-function-scorer
update-lambda-function-scorer:
	aws s3 cp --profile production dist/scorer_lambda.pex s3://io.etherlabs.artifacts/${ENV}/scorer_lambda.pex
	aws lambda update-function-code --function-name pim --s3-bucket io.etherlabs.artifacts --s3-key ${ENV}/scorer_lambda.pex

.PHONY: update-lambda-function-gs
update-lambda-function-gs:
	aws s3 cp --profile ${ENV} dist/group_segments_code.pex s3://io.etherlabs.artifacts/${ENV}/group_segments_code.pex
	aws lambda update-function-code --function-name group-segments --s3-bucket io.etherlabs.artifacts --s3-key ${ENV}/group_segments_code.pex

.PHONY: test-gs
test-gs:
	aws s3 cp --profile staging2 dist/group_segments_code.pex s3://io.etherlabs.staging2.contexts/topics/group_segments_code.pex
	aws lambda update-function-code --function-name pex_test --s3-bucket io.etherlabs.staging2.contexts --s3-key topics/group_segments_code.pex --region=us-east-1
