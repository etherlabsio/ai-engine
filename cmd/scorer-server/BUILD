python_binary(
	name="server",
	dependencies = [
		'services/scorer:scorer-libs',
		':scorer-dep',
		'services:global-services',
        'pkg:common-libs',
	],
	source="main.py",
        compatibility=['CPython==3.7.*'],
)

python_library(
	name="scorer-dep",
	dependencies=[
		'3rdparty/python:iso8601',
        '3rdparty/python:boto3',
        '3rdparty/python:botocore',
        '3rdparty/python:numpy',
        '3rdparty/python:python-json-logger',
        '3rdparty/python:nltk',
   ],              

)


python_awslambda(
  name='scorer_lambda',
  binary=':server',
  handler='main:handler',
)
