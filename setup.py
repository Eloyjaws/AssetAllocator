import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

DEPENDENCIES = []
with open("requirements.txt", "rb") as reqs:
    for line in reqs.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[: line.find("#")].strip()
        if line:
            DEPENDENCIES.append(line)

setuptools.setup(
    name="AssetAllocator",
    version="0.0.11",
    author="Adebayo Oshingbesan, Eniola Ajiboye, Peruth Kamashazi, Timothy Mbaka",
    author_email="eajiboye@andrew.cmu.edu",
    description="Train RL agents to manage a portfolio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eloyjaws/AssetAllocator",
    packages=setuptools.find_packages(),
    license="MIT",
    install_requires=DEPENDENCIES,
    package_dir={'AssetAllocator': 'AssetAllocator'},
    package_data={'AssetAllocator': ['data/*']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
