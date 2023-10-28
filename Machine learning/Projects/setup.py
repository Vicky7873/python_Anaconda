from setuptools import setup,find_packages
from typing import List

def get_requirements(file_path:str)-> List[str]:
    # make a function to read library from requiemetn.txt file
    requirement=[]
    with open(file_path) as file_obj:
        requirement=file_obj.readlines()
        requirement=[req.replace("\n","") for req in requirement]

        return requirement

setup(
    name="DiamondPricePrediction",
    version='0.0.1',
    author='bhiki',
    author_email='vicky.pallai900@gmai.com',
    install_requires=get_requirements('requirement.txt'),
    packages=find_packages()
)