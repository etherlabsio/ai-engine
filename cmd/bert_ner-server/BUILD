python_binary(
	name="server",
	dependencies = [
		'pkg:common-libs',
		'services/bert_ner:bert_ner-libs',
		':bert_ner-dep',
		'services:bert_ner',
	],
	source="main.py",

    compatibility=['CPython==3.7.*'],

    platforms=["linux-x86_64"]
)

python_library(
	name="bert_ner-dep",
	dependencies=[
		'3rdparty/python:boto3',
        '3rdparty/python:botocore',
        '3rdparty/python:numpy',
        '3rdparty/python:python-json-logger',
        '3rdparty/python:requests',
        '3rdparty/python:nltk',
	],

)

python_awslambda(
name='ner_lambda',
binary=':server',
handler='main:handler',
)
