import setuptools

setuptools.setup(
    name='raman',
    author='Bryan Fichera',
    author_email='bfichera@mit.edu',
    description='A collection of utilities for analyzing Raman data.',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'sympy',
        'polars',
    ]
)
