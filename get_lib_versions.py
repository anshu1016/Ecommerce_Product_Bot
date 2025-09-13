import importlib.metadata
import os
from pprint import pprint



packages = []

with open('requirements.txt', 'r') as f:
    for line in f:
        package = line.split('==')[0].strip()
        packages.append(package)


for i, package in enumerate(packages):
    try:
        version = importlib.metadata.version(package)
        packages[i] = f"{package}=={version}"
    except importlib.metadata.PackageNotFoundError:
        packages[i] = f"{package} (not installed)"

pprint(packages)
