python_binary(
	name="server",
	dependencies = [
        'pkg:distil_bilstm_pos_tagger-libs',
        'pkg:s3-libs',
        'pkg:logger-libs',
		'services/action_item_detector:action_item_detector-libs',
		':action_item_detector-dep',
	],
	source="main.py",
    compatibility=['CPython==3.7.*'],
    platforms=['linux-x86_64']
)

python_library(
	name="action_item_detector-dep",
	dependencies=[
        '3rdparty/python:boto3',
        '3rdparty/python:botocore',
        '3rdparty/python:numpy',
        '3rdparty/python:python-json-logger',
        '3rdparty/python:requests',
        '3rdparty/python:nltk',
        '3rdparty/python:uuid',
        ]
)

python_awslambda(
  name='action_item_detection-lambda',
  binary=':server',
  handler='main:handler',
)
