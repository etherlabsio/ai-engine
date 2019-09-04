python_binary(
	name="sentencoder-server",
	dependencies = [
		'pkg:common-libs',
		':sentencoder-dep',
		'services/sentence_encoder:sentencoder-libs',
		'services:global-services'
	],
	source="main.py",

    compatibility=['CPython==3.7.*'],
)

python_library(
	name="sentencoder-dep",
	dependencies=[
        '3rdparty/python:python-json-logger',
		'3rdparty/python:boto3',
		'3rdparty/python:tensorflow-hub',
		'3rdparty/python:numpy'
	],

)

python_awslambda(
  name='sentencoder-lambda',
  binary=':sentencoder-server',
  handler='main:lambda_handler',
)
