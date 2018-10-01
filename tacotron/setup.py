#!/usr/bin/env python

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.build_py
import os
import subprocess

version = '0.1.1'

# Adapted from https://github.com/pytorch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv('TACOTRON2_BUILD_VERSION'):
    version = os.getenv('TACOTRON2_BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except subprocess.CalledProcessError:
        pass
    except IOError:  # FileNotFoundError for python 3
        pass


class build_py(setuptools.command.build_py.build_py):

    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global version, cwd
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))


class develop(setuptools.command.develop.develop):

    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


setup(name='tacotron2',
      version=version,
      description='PyTorch implementation of NVIDIA Tactron2',
      packages=find_packages(),
      cmdclass={
          'build_py': build_py,
          'develop': develop,
      },
      install_requires=[
          "numpy == 1.13.3",
          "scipy == 1.0.0",
          "torch >= 0.4.0",
          "matplotlib == 2.1.0",
          "tensorflow == 1.6.0",
          "numpy == 1.13.3",
          "inflect == 0.2.5",
          "librosa == 0.6.0",
          "tensorboardX == 1.1",
          "Unidecode == 1.0.22",
          "pillow",
      ],
      extras_require={
          "train": [
              "docopt",
              "tqdm",
              "tensorboardX == 1.1",
              "nnmnkwii >= 0.0.11",
              "keras",
              "scikit-learn",
              "lws <= 1.0",
          ],
          "test": [
              "nose",
              "pysptk >= 0.1.9",
              "tqdm",
              "nnmnkwii >= 0.0.11",
          ],
      })
