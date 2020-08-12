## [Monk - A computer vision toolkit for everyone](https://monkai.org/) [![Tweet](https://img.shields.io/twitter/url/https/github.com/tterb/hyde.svg?style=social)](http://twitter.com/share?text=Check%20out%20Monk:%20An%20Open%20Source%20Unified%20Wrapper%20for%20Computer%20Vision&url=https://github.com/Tessellate-Imaging/monk_v1&hashtags=MonkAI,OpenSource,UnifiedWrapper,DeepLEarning,ComputerVision,TessellateImaging) [![](http://hits.dwyl.io/Tessellate-Imaging/monk_v1.svg)](http://hits.dwyl.io/Tessellate-Imaging/monk_v1)  ![](https://tokei.rs/b1/github/Tessellate-Imaging/monk_v1) ![](https://tokei.rs/b1/github/Tessellate-Imaging/monk_v1?category=files) 
[![Version](https://img.shields.io/badge/version-v0.0.1-darkgreen)](https://github.com/Tessellate-Imaging/monk_v1) [![Build_Status](https://img.shields.io/badge/build-passing-darkgreen)](https://github.com/Tessellate-Imaging/monk_v1)
 
<br />


### Why use Monk
 - Issue: Want to begin learning computer vision
   - <b> Solution: Start with Monk's hands-on study roadmap tutorials</b>
   
 - Issue: Multiple libraries hence multiple syntaxes to learn
   - <b> Solution: Monk's one syntax to rule them all - pytorch, keras, mxnet, etc </b>
 
 - Issue: Tough to keep track of all the trial projects while participating in a deep learning competition
   - <b> Solution: Use monk's project management and work on multiple prototyping experiments</b>
 
 - Issue: Tough to set hyper-parameters while training a classifier
   - <b> Solution: Try out hyper-parameter analyser to find the right fit </b>
 
 - Issue: Looking for a library to build quick solutions for your customer
   - <b> Solution: Train, Infer and deploy with monk's low-code syntax </b>
   
<br />
<br />


## Create real-world Image Classification applications 
<table>
  <tr>
    <td>Medical Domain</td>
    <td>Fashion Domain</td>
    <td>Autonomous Vehicles Domain</td>
  </tr>
  <tr>
    <td><img src="https://github.com/abhi-kumar/monk_cls_demos/blob/master/cls-chest-xray-pneumonia-demo.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_cls_demos/blob/master/cls-apparel-demo.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_cls_demos/blob/master/cls-distracted-driver-demo.gif" width=320 height=240></td>
  </tr>
  <tr>
    <td>Agriculture Domain</td>
    <td>Wildlife Domain</td>
    <td>Retail Domain</td>
  </tr>
  <tr>
    <td><img src="https://github.com/abhi-kumar/monk_cls_demos/blob/master/cls-rice-leaf-disease-demo.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_cls_demos/blob/master/cls-oregon-wildlife-species-demo.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_cls_demos/blob/master/cls-groceries-demo.gif" width=320 height=240></td>
  </tr>
  <tr>
    <td>Satellite Domain</td>
    <td>Healthcare Domain</td>
    <td>Activity Analysis Domain</td>
  </tr>
    <td><img src="https://github.com/abhi-kumar/monk_cls_demos/blob/master/cls-land-usage-demo.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_cls_demos/blob/master/cls-mask-demo.gif" width=320 height=240></td>
    <td><img src="https://github.com/abhi-kumar/monk_cls_demos/blob/master/cls-yoga82-demo.gif" width=320 height=240></td>
 </table>

### ...... For more check out the [Application Model Zoo](https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/4_image_classification_zoo)!!!! 

<br />
<br />
 
## How does Monk make image classification easy
 - Write **less code** and create end to end applications.
 - Learn only **one syntax** and create applications using any deep learning library - pytorch, mxnet, keras, tensorflow, etc
 - Manage your entire project easily with multiple experiments

<br />
<br />

## For whom this library is built
  - **Students**
    - Seamlessly learn computer vision using our comprehensive study roadmaps
  - **Researchers and Developers**
    - Create and Manage multiple deep learning projects 
  - **Competiton participants** (Kaggle, Codalab, Hackerearth, AiCrowd, etc)
    - Expedite the prototyping process and jumpstart with a higher rank
    
<br />
<br />


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
## Sample Showcase - Quick Mode

#### Create an image classifier.
```python
#Create an experiment
ptf.Prototype("sample-project-1", "sample-experiment-1")

#Load Data
ptf.Default(dataset_path="sample_dataset/", 
             model_name="resnet18", 
             num_epochs=2)
# Train
ptf.Train()
```

#### Inference

```python
predictions = ptf.Infer(img_name="sample.png", return_raw=True);
```


#### Compare Experiments

```python
#Create comparison project
ctf.Comparison("Sample-Comparison-1");

#Add all your experiments
ctf.Add_Experiment("sample-project-1", "sample-experiment-1");
ctf.Add_Experiment("sample-project-1", "sample-experiment-2");
   
# Generate statistics
ctf.Generate_Statistics();
```

<br />
<br />
<br />


<a id="2"></a>
## Installation
  - **CUDA 9.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-cuda90`
  - **CUDA 9.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-cuda92`
  - **CUDA 10.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-cuda100`
  - **CUDA 10.1** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-cuda101`
  - **CUDA 10.2** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-cuda102`
  - **CPU** (+Mac-OS)&nbsp;: `pip install -U monk-cpu`
  - **Google Colab** &nbsp;&nbsp;: `pip install -U monk-colab`
  - **Kaggle** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-kaggle`
  
For More Installation instructions visit: [Link](https://github.com/Tessellate-Imaging/monk_v1/tree/master/installation)


<br />
<br />
<br />


<a id="3"></a>
## Study Roadmaps


  - [Getting started with Monk](https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/1_getting_started_roadmap)
    - Essential notebooks to use all the monk's features
  - [Image Processing and Deep Learning](https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/3_image_processing_deep_learning_roadmap)
    - Learn both the basic and advanced concepts of image processing and deep learning
  - [Transfer Learning](https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/2_transfer_learning_roadmap)
    - Understand transfer learning in the AI field
  - [Image classification zoo](https://github.com/Tessellate-Imaging/monk_v1/tree/master/study_roadmaps/4_image_classification_zoo)
    - A list of 50+ real world image classification examples 


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

### Features
- [x] Model Visualization
- [ ] Pre-processed data visualization
- [x] Learned feature visualization
- [ ] NDimensional data input - npy - hdf5 - dicom - tiff
- [x] Multi-label Image Classification
- [x] Custom model development



### General
- [x] Functional Documentation
- [x] Tackle Multiple versions of libraries
- [x] Add unit-testing
- [ ] Contribution guidelines
- [x] Python pip packaging support


### Backend Support
- [x] Tensorflow 2.0 provision support with v1
- [ ] Tensorflow 2.0 complete
- [ ] Chainer


### External Libraries
- [ ] TensorRT Acceleration
- [ ] Intel Acceleration
- [ ] Echo AI - for Activation functions

<br />
<br />

### Connect with the project [contributors](https://github.com/Tessellate-Imaging/monk_v1/blob/master/Contributors.md)

<br />
<br />

## Copyright

Copyright 2019 onwards, Tessellate Imaging Private Limited Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.
