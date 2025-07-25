from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    
    try:
    
        with open("requirements.txt", "r") as f:
            reqs = f.readlines()
            reqs = [package.replace("\n","").strip() for package in reqs]
            
            if "-e ." in reqs:
                reqs.remove("-e .")
                
                
    except FileNotFoundError:
        print("requirements.txt is not found")
        
    return reqs

setup(
    name="recipe-site-traffic-classification",
    version="0.0.1",
    description="Recipe Site Traffic Classification Project",
    author="Richard Holzhofer",
    author_email="richard.holzhofer@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)