python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
		'services/keyphrase:keyphrase-libs',
		':keyphrase-dep',
	],
	source="main.py",
	compatibility="Cpython>3",
)

python_library(
	name="keyphrase-dep",
	dependencies=[
		'3rdparty/python:asyncio-nats-client',
		'3rdparty/python:textblob',
		'3rdparty/python:tornado',
		'3rdparty/python:sanic',
		'3rdparty/python:python-dotenv',
		'3rdparty/python:langdetect',
		'3rdparty/python:six',
		'3rdparty/python:structlog',
		'3rdparty/python:spacy',
		'3rdparty/python:pandas',
		'3rdparty/python:pydgraph',
		'3rdparty/python:iso8601',
		'3rdparty/python:sklearn',
		'3rdparty/python:nltk',
		'3rdparty/python:networkx',
		'3rdparty/python:pylint',
		'3rdparty/python:autopep8',
		'3rdparty/python:rope',
	],

)
