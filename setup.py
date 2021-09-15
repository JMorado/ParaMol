from setuptools import setup, find_packages
setup(
    name='ParaMol',
    version='1.1.4',
    packages=find_packages(),
    url='https://github.com/JMorado/ParaMol',
    license='MIT',
    author='João Morado',
    author_email='j.morado@soton.ac.uk',
    description='A Package for Parametrization of Molecular Mechanics Force Fields',
    tests_require=["pytest"],
    test_suite="ParaMol.Tests",
)
