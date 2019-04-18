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

COPY services/keyphrase/Pipfile Pipfile
COPY services/keyphrase/Pipfile.lock Pipfile.lock

RUN pip install virtualenv --upgrade
RUN pip install pipenv --upgrade
RUN pip install wheel --upgrade
RUN set -ex && pipenv --python 3.7
RUN pipenv install -v --system
RUN python3.7 -m spacy download en_core_web_sm && python3.7 -m spacy download en

# Download NLTK data
RUN python3.7 -c "import nltk; nltk.download('punkt')"
RUN python3.7 -c "import nltk; nltk.download('stopwords')"
RUN python3.7 -c "import nltk; nltk.download('averaged_perceptron_tagger')"

COPY services/ /services/
COPY pkg /common/

ENTRYPOINT ["python", "services/keyphrase/main.py"]
CMD []
# EXPOSE 8080
