python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
		'services/ether_graph:ether_graph_libs',
		':ether_graph_dep',
		'services:global-services'
	],
	source="main.py",

    compatibility=['CPython==3.7.*'],
)

python_library(
	name="ether_graph_dep",
	dependencies=[
		'3rdparty/python:asyncio-nats-client',
		'3rdparty/python:uvloop',
        '3rdparty/python:python-json-logger',
		'3rdparty/python:python-dotenv',
		'3rdparty/python:pydgraph',
		'3rdparty/python:dataclasses-json',
		'3rdparty/python:ciso8601'
	],

)
