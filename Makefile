AWS_ACCOUNT_ID=$(shell aws --profile ${ENV} sts get-caller-identity --output text --query 'Account')
AWS_ACCESS_KEY_ID=$(shell aws configure get aws_access_key_id --profile ${ENV})
AWS_SECRET_ACCESS_KEY=$(shell aws configure get aws_secret_access_key --profile ${ENV})
AWS_REGION=$(shell aws configure get region --profile ${ENV})
LOGIN=$(shell AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} aws ecr get-login --no-include-email --region ${AWS_REGION})

IMAGE_PREFIX=${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/etherlabs
CONTAINER_IMAGE_NEW=${IMAGE_PREFIX}/{app}

ENV=staging2
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"
BRANCH=$(shell git rev-parse --short HEAD || echo -e '$CI_COMMIT_SHA')


pre-deploy-notify:
	@curl -X POST --data-urlencode 'payload={"text": "[${ENV}] [${BRANCH}] ${USER}: ${ARTIFACT} is being deployed"}' \
				 ${SLACK_WEBHOOK_URL}

post-deploy-notify:
	@curl -X POST --data-urlencode 'payload={"text": "[${ENV}] [${BRANCH}] ${USER}: ${ARTIFACT} is deployed"}' \
				 ${SLACK_WEBHOOK_URL}

docker_login:
	@eval ${LOGIN}


build: docker_login
	echo ${ENV} ${ARTIFACT} ${CONTAINER_IMAGE} ${BRANCH}
	@docker build --build-arg app=${APP} -t ${CONTAINER_IMAGE}:${BRANCH} .

docker_push:
	@docker push ${CONTAINER_IMAGE}:${BRANCH}

ifeq (${ENV},production)
	@docker tag ${CONTAINER_IMAGE}:${BRANCH} ${CONTAINER_IMAGE}:latest
	@docker push ${CONTAINER_IMAGE}:latest
else
	@docker tag ${CONTAINER_IMAGE}:${BRANCH} ${CONTAINER_IMAGE}:${ENV}
	@docker push ${CONTAINER_IMAGE}:${ENV}
endif

deploy_ecs: build docker_push
	$(MAKE) pre-deploy-notify
	ecs deploy --region ${AWS_REGION} --access-key-id ${AWS_ACCESS_KEY_ID} --secret-access-key ${AWS_SECRET_ACCESS_KEY} ${CLUSTER_NAME} ${SERVICE_NAME} --task ${FAMILY} --tag ${BRANCH} --timeout 600
	$(MAKE) post-deploy-notify

###############
# Deployments #
###############

# Keyphrase
deploy_keyphrase:
	$(MAKE) deploy_ecs APP=keyphrase ARTIFACT=keyphrase-server CONTAINER_IMAGE=${IMAGE_PREFIX}/keyphrase \
		 CLUSTER_NAME=ml-inference SERVICE_NAME=keyphrase-service FAMILY=keyphrase

# Recommendation
deploy_recommendation:
	$(MAKE) deploy_ecs APP=recommendation ARTIFACT=recommendation-server CONTAINER_IMAGE=${IMAGE_PREFIX}/recommendation \
		 CLUSTER_NAME=ml-inference SERVICE_NAME=recommendation-service FAMILY=recommendation


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
	aws s3 cp --profile ${ENV} dist/group_segments_code.pex s3://io.etherlabs.artifacts/${ENV}/group_segments_code.pex
	aws lambda update-function-code --function-name group-segments --s3-bucket io.etherlabs.artifacts --s3-key ${ENV}/group_segments_code.pex

.PHONY: build-upload-update-lambda
build-upload-update-lambda:
	./pants bundle cmd/${app-name}-server:${build-name}
	aws s3 cp --profile ${ENV} dist/${build-name}.pex s3://io.etherlabs.artifacts/${ENV}/${build-name}.pex
	aws lambda update-function-code --function-name ${function-name} --s3-bucket io.etherlabs.artifacts --s3-key ${ENV}/${build-name}.pex
