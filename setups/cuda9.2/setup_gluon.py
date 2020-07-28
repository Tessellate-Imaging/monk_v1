import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="monk_gluon_cuda92", # Replace with your own username
    version="0.0.1",
    author="Tessellate Imaging",
    author_email="abhishek@tessellateimaging.com",
    description="Monk Classification Library - Cuda92 - backends - mxnet-gluon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tessellate-Imaging/monk_v1",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA :: 9.2",
    ],
    install_requires=[
        'scipy',
        'scikit-learn',
        'scikit-image',
        'opencv-python',
        'pillow==6.0.0',
        'tqdm',
        'gpustat', 
        'psutil',
        'pandas',
        'GPUtil',
        'mxnet-cu92==1.5.1',
        'gluoncv==0.6',
        'torch==1.4.0',
        'tabulate',
        'netron',
        'networkx',
        'matplotlib',
        'pylg',
        'ipywidgets'
    ],
    python_requires='>=3.6',

)





