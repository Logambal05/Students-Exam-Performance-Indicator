from setuptools import find_packages, setup
from typing import List

def Get_Requirement(file_path: str) -> List:
    """This Function Returns the Required packages as a List"""
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements

setup(
    name="ML Project",
    version="0.0.1",
    author="Logambal",
    author_email="logi2987@gmail.com",
    packages=find_packages(),
    install_requires=Get_Requirement("Requirements.txt")
)