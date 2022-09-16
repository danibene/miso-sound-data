#!/usr/bin/python3

from setuptools import setup

setup(
    name="misosoundbank",
    version="0.1.0",
    description="API for an open-access sound bank intended for misophonia research",
    long_description=",
    author="Danielle Benesch",
    author_email="dbenesch@critias.ca",
    py_modules=["misosoundbank"],
    install_requires=["numpy",
                      "pandas",
                      "wget",
                      "librosa",
                      "pydub",
                      "soundfile"],
    zip_safe=False,
    url="https://github.com/miso-sound/miso-sound-data",
    license="MIT",
)
