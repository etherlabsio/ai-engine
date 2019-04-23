FROM derphilipp/ubuntu_bionic_with_utf8

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive  apt-get -y install --no-install-recommends  build-essential \
        python3.7 \
        python3-pip python3.7-dev && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/archives && \
    rm -f /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/bin/python && \
    ln -s /usr/bin/python3.7 /usr/bin/python3 && \
        python3.7 -m pip install --upgrade pip

RUN mkdir /opt/app
WORKDIR /opt/app

COPY pkg pkg/
COPY dist/keyphrase-bin.pex keyphrase-bin.pex
ENTRYPOINT ["./keyphrase-bin.pex"]
CMD []
# EXPOSE 8080
