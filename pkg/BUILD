python_library(
    name='common-libs',
    sources=['*/*', '*']
)

python_library(
    name='graphrank-pkg',
    sources=['graphrank/*.py'],
    dependencies=[
        'pkg/graphrank:graphrank'
    ],
    provides=setup_py(
        name='graphrank',
        version='0.5.5',
        description='Package to perform unsupervised keyphrase extraction using graphs.',
        author='Shashank',
        author_email='shashank@etherlabs.io',
        python_requires='>=3.6.0',
        url='https://github.com/etherlabsio/ai-engine',
        license='MIT',
        classifiers=[
            # Trove classifiers
            # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython',
            'Programming Language :: Python :: Implementation :: PyPy'
        ],
    )
)


python_library(
    name='text_preprocessing-pkg',
    sources=['text_preprocessing/*.py'],
    dependencies=[
        'pkg/text_preprocessing:text_preprocessing',
        '3rdparty/python:nltk',
    ],
    provides=setup_py(
        name='text_preprocessing',
        version='0.2',
        description='Text pre-processing package for almost every NLP task',
        author='Reagon Rewop',
        author_email='sri@etherlabs.io',
        python_requires='>=3.6.0',
        url='https://github.com/etherlabsio/ai-engine',
        license='MIT',
        classifiers=[
            # Trove classifiers
            # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython',
            'Programming Language :: Python :: Implementation :: PyPy'
        ],
    )
)
