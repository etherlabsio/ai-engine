python_library(
    name='distilbert_pos_tagger',
    sources=["*"],
    dependencies=[
        ':distilbert_utils',
		'3rdparty/python:nltk',
		'3rdparty/python:torch',
		'3rdparty/python:boto3',
		'3rdparty/python:requests',
		'3rdparty/python:numpy'
    ],
    compatibility=['CPython>=3.6.*']
)

python_library(
    name='distilbert_utils',
    sources=["distilbert_utils/*.py"]
)

