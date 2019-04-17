FROM kennethreitz/pipenv

WORKDIR /app

# Install additional dependencies
RUN python3 -m spacy download en_core_web_sm && \
    python3 -m spacy download en && \
    python3 -c "import nltk; nltk.download('punkt')" && \
    python3 -c "import nltk; nltk.download('stopwords')" && \
    python3 -c "import nltk; nltk.download('averaged_perceptron_tagger')"

COPY . .

ENTRYPOINT ["python3", "keyphrase/main.py"]
CMD []
