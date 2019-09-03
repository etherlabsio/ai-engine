python_binary(
	name="sentencoder-server",
	dependencies = [
		'pkg:common-libs',
		':sentencoder-dep',
		'services/sentence_encoder:sentencoder-libs',
	],
	source="main.py",

    compatibility=['CPython==3.7.*'],
)

python_library(
	name="sentencoder-dep",
	dependencies=[
        '3rdparty/python:python-json-logger',
		'3rdparty/python:python-dotenv',
		'3rdparty/python:boto3',
		'3rdparty/python:tensorflow',
		'3rdparty/python:tensorflow-hub'

	],

)

python_awslambda(
  name='sentencoder-lambda',
  binary=':sentencoder-server',
  handler='main:lambda_handler',
)
