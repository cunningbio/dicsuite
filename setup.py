from setuptools import setup, find_packages

setup(
    name='dicsuite',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'matplotlib',
        'pillow'
    ],
    extras_require={
        'gpu': ['cupy-cuda12x']  # user installs correct CUDA version
    },
    author='Richard Cunningham', author_email="rcunnin2@ed.ac.uk",
    description='DIC-to-QPI Reconstruction Toolkit with CPU/GPU support',
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: BSD License",
        'Operating System :: OS Independent',
    ],
)