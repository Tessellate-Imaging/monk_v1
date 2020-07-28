import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="monk_colab", # Replace with your own username
    version="0.0.1",
    author="Tessellate Imaging",
    author_email="abhishek@tessellateimaging.com",
    description="Monk Classification Library - Colab - backends - pytorch, keras, gluon",
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
        'GPUtil',
        'mxnet-cu101',
        'pylg',
        'gluoncv'
    ],
    python_requires='>=3.6',

)





