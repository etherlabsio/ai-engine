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

.PHONY: run
run:
	./pants run cmd/${app}-server:server

.PHONY: binary
binary:
	./pants binary cmd/${app}-server:server

.PHONY: docker-build
docker-build:
	sudo docker build . --tag ${app} --build-arg app=${app}

.PHONY: docker-run
docker-run:
	sudo docker run -p 8080:7070 ${app}

.PHONY: new-service
new-service:
	@mkdir services/${app}
	@mkdir cmd/${app}-server
	@touch cmd/${app}-server/main.py
	@cp .template/BUILD.cmd cmd/${app}-server/BUILD
	@cp .template/BUILD.services services/${app}/BUILD
	@echo -e '\n\n Added cmd/${app}-server with BUILD & main file \n Added services/${app} with BUILD file'
	@echo -e '\nNote: Kindly go into the Build files present in the 'services/${app}/' and 'cmd/${app}-server/'. \n Change the service name from Keyphrase to your respective service name and, \n add/remove the dependencies mentioned in the cmd/${app}-server/BUILD file '