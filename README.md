# monk_v1 [![Tweet](https://img.shields.io/twitter/url/https/github.com/tterb/hyde.svg?style=social)](http://twitter.com/share?text=Check%20out%20Monk:%20An%20Open%20Source%20Unified%20Wrapper%20for%20Computer%20Vision&url=https://github.com/Tessellate-Imaging/monk_v1&hashtags=MonkAI,OpenSource,UnifiedWrapper,DeepLEarning,ComputerVision,TessellateImaging) [![](http://hits.dwyl.io/Tessellate-Imaging/monk_v1.svg)](http://hits.dwyl.io/Tessellate-Imaging/monk_v1)  ![](https://tokei.rs/b1/github/Tessellate-Imaging/monk_v1) ![](https://tokei.rs/b1/github/Tessellate-Imaging/monk_v1?category=files)


[Website](https://monkai.org/)

#### Monk is a low code Deep Learning tool and a unified wrapper for Computer Vision.
[![Version](https://img.shields.io/badge/version-v1.0-lightgrey)](https://github.com/Tessellate-Imaging/monk_v1) &nbsp; &nbsp;
[![Build_Status](https://img.shields.io/badge/build-passing-green)](https://github.com/Tessellate-Imaging/monk_v1)


# Table of Contents
  - [Sample Showcase](#1)
  - [Installation](#2)
  - [Study Roadmaps, Examples, and Tutorials](#3)
  - [Documentation](#4)
  - [TODO](#5)

<br />
<br />
<br />


<a id="1"></a>
## Sample Showcase

#### Create an image classification experiment.
- Load foldered dataset
- Set number of epochs
- Run training

```python
ptf = prototype(verbose=1)
ptf.Prototype("sample-project-1", "sample-experiment-1")
ptf.Default(dataset_path="./dataset_cats_dogs_train/", 
                model_name="resnet18", freeze_base_network=True, num_epochs=2)
ptf.Train()
```

#### Inference

```python
img_name = "./monk/datasets/test/0.jpg";
predictions = ptf.Infer(img_name=img_name, return_raw=True);
print(predictions)
```


#### Compare Experiments

- Add created experiments with different hyperparameters
- Generate comparison plots

```python
ctf = compare(verbose=1);
ctf.Comparison("Sample-Comparison-1");
ctf.Add_Experiment("sample-project-1", "sample-experiment-1");
ctf.Add_Experiment("sample-project-1", "sample-experiment-2");
    .
    . 
    .
ctf.Generate_Statistics();
```

<br />
<br />
<br />

<a id="2"></a>
## Installation

Support for
  - OS
      - Ubuntu 16.04
      - Ubuntu 18.04
      - Mac OS
      - Windows 
  - Python
      - Version 3.6
      - Version 3.7
  - Cuda
      - Version 9.0
      - Version 9.2
      - Version 10.0
      - Version 10.1

For Installation instructions visit: [Link](https://github.com/Tessellate-Imaging/monk_v1/tree/master/installation)


<br />
<br />
<br />


<a id="3"></a>
## Study Roadmaps


  - [Getting started with Monk](https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/1_getting_started_roadmap)
  - [Python sample examples](https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/python_sample_examples)
  - [Image Processing and Deep Learning](https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/3_image_processing_deep_learning_roadmap)
  - [Transfer Learning](https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/2_transfer_learning_roadmap)
  - [Image classification zoo](https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/4_image_classification_zoo)


<br />
<br />
<br />

<a id="4"></a>
## Documentation

- [List of available models, layers, blocks, optimizers](https://github.com/Tessellate-Imaging/monk_v1/tree/master/monk)

- [Functional Documentation](https://abhi-kumar.github.io/monk_v1_docs/) (Will be merged with Latest docs soon)
    - Main Prototype Functions
        - [Mxnet Backend](https://abhi-kumar.github.io/monk_v1_docs/gluon_prototype.html)
        - [Pytorch Backend](https://abhi-kumar.github.io/monk_v1_docs/pytorch_prototype.html)
        - [Keras Backend](https://abhi-kumar.github.io/monk_v1_docs/keras_prototype.html)
        - [Comparison](https://abhi-kumar.github.io/monk_v1_docs/compare_prototype.html)
    - [System Functions](https://abhi-kumar.github.io/monk_v1_docs/system/index.html)
    - [Mxnet Backend base Functions](https://abhi-kumar.github.io/monk_v1_docs/gluon/index.html)
    - [Pytorch Backend base Functions](https://abhi-kumar.github.io/monk_v1_docs/pytorch/index.html)
    - [Keras Backend base Funtions](https://abhi-kumar.github.io/monk_v1_docs/tf_keras_1/index.html)

- Features and Functions (In development):
    - [Introduction](https://clever-noyce-f9d43f.netlify.com/#/introduction)
    - [Setup](https://clever-noyce-f9d43f.netlify.com/#/setup/setup)
    - [Quick Mode](https://clever-noyce-f9d43f.netlify.com/#/quick_mode/quickmode_pytorch)
    - [Update Mode](https://clever-noyce-f9d43f.netlify.com/#/update_mode/update_dataset)
    - [Expert Mode](https://clever-noyce-f9d43f.netlify.com/#/expert_mode)
    - [Hyper Parameter Analyser](https://clever-noyce-f9d43f.netlify.com/#/hp_finder/model_finder)
    - [Compare Experiments](https://clever-noyce-f9d43f.netlify.com/#/compare_experiment)
    - [Resume Training](https://clever-noyce-f9d43f.netlify.com/#/resume_training)

- [Complete Latest Docs](https://li8bot.github.io/monkai/#/home) (In Progress)


<br />
<br />
<br />


<a id="5"></a>
## TODO-2020

### TODO-2020 - Features
- [x] Model Visualization
- [ ] Pre-processed data visualization
- [ ] Learned feature visualization
- [ ] NDimensional data input - npy - hdf5 - dicom - tiff
- [x] Multi-label Image Classification
- [x] Custom model development



### TODO-2020 - General
- [ ] Incorporate pep coding standards
- [x] Functional Documentation
- [x] Tackle Multiple versions of libraries
- [x] Add unit-testing
- [ ] Contribution guidelines


### TODO-2020 - Backend Support

- [ ] Tensorflow 2.0
- [ ] Chainer


### TODO-2020 - External Libraries
- [ ] TensorRT Acceleration
- [ ] Intel Acceleration
- [ ] Echo AI - for Activation functions




## Copyright

Copyright 2019 onwards, Tessellate Imaging Private Limited Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.
