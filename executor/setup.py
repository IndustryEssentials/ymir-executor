""" Setup
"""
from setuptools import setup, find_packages
from os import path

long_description = "blabla"

setup(
    name='executor',
    version='1.0.0',
    description='ymir executor',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yzbx/ymir-executor@executor',
    author='youdaoyzbx',
    author_email='youdaoyzbx@163.com',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='ymir docker executor',
    packages=find_packages(exclude=['app', 'tests']),
    include_package_data=True,
    # install_requires=['pydantic>=1.8.2', 'pyyaml>=5.4.1'],
    python_requires='>=3.6',
)