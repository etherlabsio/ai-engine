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
	sudo docker build -t ${DOCKER_IMAGE}:staging . --build-arg app=${app}
	docker push ${DOCKER_IMAGE}:staging

.PHONY: staging2
staging2: version
	sudo docker build -t ${DOCKER_IMAGE}:staging2 . --build-arg app=${app}
	docker push ${DOCKER_IMAGE}:staging2

.PHONY: production
production: version
	sudo docker build -t ${DOCKER_IMAGE} . --build-arg app=${app}
	docker push ${DOCKER_IMAGE}

.PHONY: clean
clean:
	rm -f .version
	docker system prune -f

.PHONY: deploy-staging
deploy-staging:
	sup -f Deployfile staging deploy

.PHONY: deploy-staging2
deploy-staging2:
	ecs deploy ml-inference keyphrase-service --timeout 600 --profile staging2

.PHONY: deploy-production
deploy-production:
	ecs deploy ml-inference keyphrase-service --timeout 600 

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
