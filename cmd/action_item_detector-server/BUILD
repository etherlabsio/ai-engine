python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
		'services/action_item_detector:action_item_detector-libs',
		':action_item_detector-dep',
		'services:global-services',
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
        ]
)

python_awslambda(
  name='action_item_detector-lambda',
  binary=':action_item_detector-server',
  handler='main:handler',
)