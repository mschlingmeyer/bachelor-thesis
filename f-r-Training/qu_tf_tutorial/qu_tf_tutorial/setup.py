from setuptools import setup, find_packages

setup(
    name='qu_tf_tutorial',
    version='0.0.1',
    install_requires=[
        """tensorflow>=2.3.0
protobuf>=3.12.4
scikit-learn>=0.23.2
pandas>=1.1.0
wget>=3.2
matplotlib>=3.3.0
tqdm>=4.48.2
livelossplot>=0.5.3
jupyter_contrib_nbextensions>=0.5.1
rise>=5.7.1
pyarrow>=6.0.1
parquet>=1.3.1
""".split()
    ],
    packages=find_packages(),
)