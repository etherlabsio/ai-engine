python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
		'services/action_item_detector:action_item_detector-libs',
		':action_item_detector-dep',
		'services:action_item_detector',
	],
	source="main.py",
    compatibility=['CPython==3.7.*'],
    platforms=['linux-x86_64']
)

python_library(
	name="action_item_detector-dep",
	dependencies=[
        '3rdparty/python:boto3',
        '3rdparty/python:botocore',
        '3rdparty/python:numpy',
        '3rdparty/python:python-json-logger',
        '3rdparty/python:requests',
        '3rdparty/python:nltk',
        '3rdparty/python:uuid',
        ]
)

python_awslambda(
  name='lambda_build',
  binary=':server',
  handler='main:handler',
)