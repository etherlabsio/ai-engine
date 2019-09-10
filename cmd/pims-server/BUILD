python_binary(
	name="server",
	dependencies = [
		'services/pims:pims-libs',
		':pims-dep',
		'services:global-services',
        'pkg:common-libs',
	],
	source="main.py",
        compatibility=['CPython==3.7.*'],
)

python_library(
	name="pims-dep",
	dependencies=[
		'3rdparty/python:iso8601',
        '3rdparty/python:boto3',
        '3rdparty/python:botocore',
        '3rdparty/python:numpy',
        '3rdparty/python:python-json-logger',
   ],              

)


python_awslambda(
  name='lambda_build',
  binary=':server',
  handler='main:handler',
)
