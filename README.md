# ai-engine

AI Engine is the monorepo for all of Ether's AI services.

## Instructions 

- **To run a service**:

Run: `docker build -t <container_name> . --build-arg app=${service_name}`

- **To install a package**:

For `GraphRank`: `sudo ./pants setup-py pkg:graphrank-pkg`

For `text_preprocessing`: `sudo ./pants setup-py pkg:text_preprocessing-pkg`

The packages are stored in `dist` folder from where they can installed using pip; for e.g:

`pip install dist/graphrank.tar.gz`