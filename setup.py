from setuptools import setup

setup(
    name='ParaMol',
    version='1.0.0',
    packages=['ParaMol', 'ParaMol.Tasks', 'ParaMol.Tasks.forces_parallel', 'ParaMol.Tests', 'ParaMol.Utils',
              'ParaMol.Utils.other_utils', 'ParaMol.System', 'ParaMol.MM_engines', 'ParaMol.Optimizers',
              'ParaMol.Optimizers.devel', 'ParaMol.QM_engines', 'ParaMol.Force_field', 'ParaMol.Parameter_space',
              'ParaMol.Objective_function', 'ParaMol.Objective_function.Properties'],
    url='https://github.com/JMorado/ParaMol',
    license='MIT',
    author='João Morado',
    author_email='j.morado@soton.ac.uk',
    description=' A Package for Parametrization of Molecular Mechanics Force Fields '
)
