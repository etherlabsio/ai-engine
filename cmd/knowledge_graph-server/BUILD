python_binary(
	name="kg-server",
	dependencies = [
		'pkg:common-libs',
		':kg-dep',
		'services/knowledge_graph_etl:kg-libs',
		'services:global-services'
	],
	source="main.py",

    compatibility=['CPython==3.7.*'],
    platforms=['linux-x86_64']
)

python_library(
	name="kg-dep",
	dependencies=[
        '3rdparty/python:python-json-logger',
		'3rdparty/python:boto3',
		'3rdparty/python:networkx'
	],

)

python_awslambda(
  name='kg-lambda',
  binary=':kg-server',
  handler='main:handler',
)


# Build for local testing

python_binary(
	name="kg-server-mac",
	dependencies = [
		'pkg:common-libs',
		':kg-dep',
		'services/knowledge_graph_etl:kg-libs',
		'services:global-services'
	],
	source="main.py",

    compatibility=['CPython==3.7.*'],
)

python_awslambda(
  name='kg-lambda-mac',
  binary=':kg-server-mac',
  handler='main:handler',
)
