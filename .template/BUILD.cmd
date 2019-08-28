python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
		'services/{app-name}:{app-name}-libs',
		':{app-name}-dep',
		'vendor:vendor-libs',
	],
	source="main.py",

    compatibility=['CPython==3.7.*'],
)

python_library(
	name="{app-name}-dep",
	dependencies=[
		'3rdparty/python:asyncio-nats-client',
		'3rdparty/python:textblob',
		'3rdparty/python:tornado',
		'3rdparty/python:sanic',
	],

)
