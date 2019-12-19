python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
        'services/scorer:scorer-libs',
		'services/segment_analyzer:segment_analyzer-libs',
		':segment_analyzer-dep',
		'services:global-services',
	],
	source="main.py",
    platforms=["linux-x86_64"],
    compatibility=['CPython==3.7.*'],
)

python_library(
	name="segment_analyzer-dep",
	dependencies=[
		'3rdparty/python:iso8601',
        '3rdparty/python:boto3',
        '3rdparty/python:botocore',
        '3rdparty/python:numpy',
        '3rdparty/python:python-json-logger',
        '3rdparty/python:nltk',
        '3rdparty/python:python-louvain',
        '3rdparty/python:networkx',
        '3rdparty/python:scikit-learn',
        '3rdparty/python:requests',
	],

)

python_awslambda(
    name='segment_analyser_lambda',
    binary=':server',
    handler='main:handler',
)   
