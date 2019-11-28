python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
		':rec-dep',
		'vendor:vendor-libs',
		'services:global-services',
		'services/recommendation:watchers-libs'
	],
	source="main.py",

    compatibility=['CPython==3.7.*'],
)

python_library(
	name="rec-dep",
	dependencies=[
		'3rdparty/python:asyncio-nats-client',
        '3rdparty/python:python-json-logger',
		'3rdparty/python:textblob',
		'3rdparty/python:langdetect',
		'3rdparty/python:six',
		'3rdparty/python:uvloop',
		'3rdparty/python:iso8601',
		'3rdparty/python:nltk',
		'3rdparty/python:boto3',
		'3rdparty/python:botocore',
		'3rdparty/python:numpy',
		'3rdparty/python:python-dotenv',
		'3rdparty/python:pydgraph',
		'3rdparty/python:jsonlines'
	],

)
