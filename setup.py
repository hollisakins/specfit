from setuptools import find_packages, setup

setup(
    name='specfit',
    packages=find_packages(include=['specfit','specfit.*']),
    version='0.1.0',
    description='spec fitter',
    author='Hollis Akins',
    license='MIT',
    install_requires=['astropy','numpy','matplotlib','scipy','tqdm'],
)
