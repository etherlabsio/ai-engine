FROM derphilipp/ubuntu_bionic_with_utf8 AS compile-image

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends build-essential python3.7 python3-pip python3.7-dev &&\
    apt-get install -y --no-install-recommends linux-headers-generic build-essential gcc python3-dev python-dev  && \
    apt-get install -y --no-install-recommends curl && \
    apt-get install -y --no-install-recommends openjdk-8-jdk ant && \
    apt-get install -y --no-install-recommends libffi-dev && \
    apt-get install -y --no-install-recommends ca-certificates-java && \
    update-ca-certificates -f;

RUN rm -rf /var/lib/apt/lists/* && \
        rm -rf /var/cache/apt/archives && \
        rm -f /usr/bin/python && \
        rm -f /usr/bin/python3 && \
        ln -s /usr/bin/python3.7 /usr/bin/python && \
        ln -s /usr/bin/python3.7 /usr/bin/python3 && \
        python3.7 -m pip install --upgrade pip

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/

WORKDIR /build
RUN curl -L -O https://pantsbuild.github.io/setup/pants && chmod +x pants
ARG app
COPY 3rdparty 3rdparty
COPY cmd cmd
COPY pkg pkg
COPY vendor vendor
COPY services services
COPY pants.ini pants.ini

RUN ./pants binary cmd/${app}-server:server

FROM python:3.7-slim

WORKDIR /app
COPY pkg pkg
COPY vendor vendor
COPY reference_prod_user.json reference_prod_user.json
COPY reference_user_kw_vector.pickle reference_user_kw_vector.pickle
COPY --from=compile-image /build/dist/server.pex .

ENV NLTK_DATA /app/vendor/nltk_data/

ENTRYPOINT ["./server.pex"]
CMD []
