python_binary(
	name="server",
	dependencies = [
		'services/minds:minds-libs',
		':minds-dep',
		'services:global-services'
	],
	source="main.py",
    compatibility=['CPython==3.7.*'],
)

python_library(
	name="minds-dep",
	dependencies=[
		'3rdparty/python:iso8601',
        '3rdparty/python:boto3',
        '3rdparty/python:botocore',
        '3rdparty/python:numpy',
        '3rdparty/python:python-json-logger',
        '3rdparty/python:requests',
        ]
)


python_awslambda(
  name='lambda_build',
  binary=':server',
  handler='main:handler',
)
