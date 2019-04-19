DOCKER_IMAGE=registry.gitlab.com/etherlabs/ether/keyphrase-server
ENV=staging

.PHONY: dependencies
dependencies:
	pipenv install

.PHONY: version
version:
	git rev-parse HEAD > .version

.PHONY: staging
staging: version
	docker build -t ${DOCKER_IMAGE}:staging .
	docker push ${DOCKER_IMAGE}:staging

.PHONY: production
production: version
	docker build -t ${DOCKER_IMAGE} .
	docker push ${DOCKER_IMAGE}

.PHONY: clean
clean:
	rm -f .version
	docker system prune -f

.PHONY: deploy-staging
deploy-staging:
	sup -f Deployfile staging deploy

.PHONY: deploy-production
deploy-production:
	sup -f Deployfile production deploy
