python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
		'services/extract_topic_pims:extract_topic_pims-libs',
		':extract_topic_pims-dep',
        'services:global-services',
	],
	source="main.py",

    compatibility=['CPython==3.7.*'],
)

python_library(
	name="extract_topic_pims-dep",
	dependencies=[
        '3rdparty/python:iso8601',  
        '3rdparty/python:python-louvain',
        '3rdparty/python:boto3',
        '3rdparty/python:botocore',
        '3rdparty/python:numpy',
        '3rdparty/python:python-json-logger',
        '3rdparty/python:nltk',
   ]
)

python_awslambda(
  name='lambda_build',
  binary=':server',
  handler='main:handler',
)


