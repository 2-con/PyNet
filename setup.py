# aufy_project/setup.py
from setuptools import setup, find_packages

setup(
  name='pynet',
  version='1.0.0',
  packages=find_packages(),
  install_requires=['numpy', 'math', 'random', 'itertools', 'time', 'matplotlib'],

  author='Aufy Mulyadi',
  author_email='72212@jisedu.or.id',
  description='PyNet is a neural network framework library for Python.',
  license='MIT',
)