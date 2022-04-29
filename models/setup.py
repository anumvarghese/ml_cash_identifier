
from setuptools import find_packages
from setuptools import setup

setup(
    name='cash_classify',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['fire==0.4.0'],
    description='cash image classifier training application.'
)
