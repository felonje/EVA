from setuptools import setup, find_packages

setup(
    name="eva-life",
    version="0.1.0",
    description="EVA — Digital Life Species. Phase A: Individual EVA.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pyyaml",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "eva-train=scripts.train:main",
            "eva-interact=scripts.interact:main",
            "eva-evaluate=scripts.evaluate:main",
            "eva-reproduce=scripts.reproduce:main",
        ],
    },
)
