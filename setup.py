#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 07.10.19
KeywordsExtractor
Extraction of keywords from a text.

:author:     Martin Dočekal
"""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open("requirements.txt") as f:
    REQUIREMENTS = f.read()

setup_args = dict(
    name='qbek',
    version='1.0.0',
    description='Extraction of keywords from a text.',
    author='Martin Dočekal',
    long_description_content_type="text/markdown",
    long_description=README,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    entry_points={
        'console_scripts': [
            'qbek = qbek.__main__:main'
        ]
    },
    keywords=['keywords extraction', 'keyphrases', 'keywords'],
    url='https://github.com/KNOT-FIT-BUT/QBEK',
    python_requires='>=3.7',
    install_requires=REQUIREMENTS.strip().split('\n'),
)

if __name__ == '__main__':
    setup(**setup_args)
