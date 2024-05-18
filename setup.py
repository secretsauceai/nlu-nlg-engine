from setuptools import setup, find_packages

setup(
    name='nlu-engine',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'datasets==2.9.0',
        'evaluate==0.4.0',
        'accelerate>=0.20.3',
        'pandas',
        'toml',
        'transformers @ git+http://github.com/huggingface/transformers.git',
    ],
)