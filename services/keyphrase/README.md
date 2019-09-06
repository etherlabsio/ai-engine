# GraphRank

Custom graph functions used for building, extending and processing word cooccurrences.
A `TextRank`-based approach (variant of `PageRank`) is used for scoring nodes and extracting keyphrases.

## Instruction to run keyphrase service

- To build the Docker image, run the following line:
`docker build -t keyphrase_service .`

- To run the container:
`docker run -v $PWD:/opt/app/ -e ACTIVE_ENV=staging --rm -it keyphrase_service --nats_url <nats://192.168.7.146:4222>`
Replace `<nats_url>` with your host's IP address. Make sure that `gnatsd` is running in another Terminal.