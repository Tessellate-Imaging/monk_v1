import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="monk_kaggle", # Replace with your own username
    version="0.0.1",
    author="Tessellate Imaging",
    author_email="abhishek@tessellateimaging.com",
    description="Monk Classification Library - Kaggle - backends - pytorch, keras, gluon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tessellate-Imaging/monk_v1",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        'scipy',
        'scikit-learn', 
        'scikit-image',
        'opencv-python', 
        'pillow==7.2.0',
        'tqdm',
        'gpustat', 
        'psutil', 
        'pandas', 
        'GPUtil', 
        'pylg',
        'tabulate',
        'netron',
        'networkx',
        'matplotlib'
    ],
    python_requires='>=3.6',

)





