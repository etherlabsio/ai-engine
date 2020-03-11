python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
		'services/artifacts_updater:artifacts_updater-libs',
		':artifacts_updater-dep',
		'services:global-services',
	],
	source="main.py",
    platforms=["linux-x86_64"],
    compatibility=['CPython==3.7.*'],
)

python_library(
	name="artifacts_updater-dep",
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
    name='artifacts_updater_lambda',
    binary=':server',
    handler='main:handler',
)   
