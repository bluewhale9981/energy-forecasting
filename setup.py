#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

readme = "This is the package for the competition Cold Start Energy Forecasting hosted by DrivenData"

requirements = [
    'jupyterlab',
    'dask',
    'numpy',
    'click',
    'catboost',
    'joblib',
    'boto3',
    'matplotlib',
    'scipy',
    'setuptools',
    'hashids',
    'seaborn',
    'bayesian-optimization',
    'python-dotenv',
    'gitpython',
    'PyYAML',
    'mlxtend',
    'tqdm',
    'tabulate',
    'termcolor',
    'rgf-python',
    'shap',
    'lightgbm',
    'xgboost==0.80',
    'urllib3',
    'requests==2.19.1',
    'google-cloud==0.33.1',
    'google-cloud-storage==1.6.0',
    'google-cloud-datastore==1.4.0',
    'category_encoders',
    'rgf-python',
    'scikit-learn==0.20.0',
    'kaggle',
    'plotly',
    'wordcloud',
    'nltk==3.3',
    'tensorflow-gpu==1.11'
]

setup_requirements = []

test_requirements = []

dependency_links = []

setup(
    author="AgilityAI",
    author_email='huy.tranquoc@asnet.com.vn',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
    description='Power Laws: Cold Start Energy Forecasting hosted by DrivenData',
    entry_points={
        'console_scripts': [
            'csef=csef.cli:main'
        ],
    },
    install_requires=requirements,
    dependency_links=dependency_links,
    license='MIT license',
    include_package_data=True,
    keywords='csef',
    name='csef',
    packages=find_packages(include=['csef.*', 'csef']),
    package_data={
        'csef': [
            "gcloud-creds.json",
            "pipeline-configs/*/*.*",
            "config/*.yml"
        ]
    },
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='git@gitlab.asoft-python.com:bgh/data-science/ml-energy-forecasting.git',
    version='1.0.0',
    zip_safe=False
)
