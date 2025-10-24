from setuptools import setup, find_packages

setup(
  name="pynet",
  version="1.0.0",
  packages=find_packages(),
  install_requires=["numpy", "math", "random", 
                    "matplotlib", "jaxlib", "jax"],

  author="Aufy Mulyadi",
  author_email="72212@jisedu.or.id",
  description="PyNet is a neural network framework and API library for Python.",
  long_description="PyNet is a neural network framework and API library for Python. It provides both a simple and a more advanced API for building, running and testing neural networks. Pynet also contains extra models and a suite of tools for data manipulation, mathematical operations, and visualization. PyNet is client-based so modified code only affects the local file and nothing else, making it suitable for experimentation and tinkering.",
  license="Apache License 2.0",
)