from setuptools import find_packages, setup
import os
from typing import List

def get_requirements(file_path:str) -> List:
    req_path=os.path.join(os.getcwd(), file_path)

    with open(req_path, 'r') as file:
        requirements=file.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements



setup(
    name='machine_learning_project',
    version='0.0.1',
    author='ErnstB',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
