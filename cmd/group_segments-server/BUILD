python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
		'services/group_segments:group_segments-libs',
		':group_segments-dep',
		'services:grouping-services',
	],
	source="main.py",
    compatibility=['CPython==3.7.*'],
)

python_library(
	name="group_segments-dep",
	dependencies=[
		'3rdparty/python:iso8601',
        '3rdparty/python:python-louvain',
        '3rdparty/python:boto3',
        '3rdparty/python:botocore',
        '3rdparty/python:numpy',
        '3rdparty/python:python-json-logger',
        '3rdparty/python:nltk',
   ],              

)


python_awslambda(
  name='lambda_build',
  binary=':server',
  handler='main:handler',
)
