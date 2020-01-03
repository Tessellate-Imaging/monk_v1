# monk_v1 [![Tweet](https://img.shields.io/twitter/url/https/github.com/tterb/hyde.svg?style=social)](http://twitter.com/share?text=Check%20out%20Monk:%20An%20Open%20Source%20Unified%20Wrapper%20for%20Computer%20Vision&url=https://github.com/Tessellate-Imaging/monk_v1&hashtags=MonkAI,OpenSource,UnifiedWrapper,DeepLEarning,ComputerVision,TessellateImaging)

[Website](https://monkai-42.firebaseapp.com/)

#### Monk is a low code Deep Learning tool and a unified wrapper for Computer Vision.
[![Version](https://img.shields.io/badge/version-v1.0-lightgrey)](https://github.com/Tessellate-Imaging/monk_v1) &nbsp; &nbsp;
[![Build_Status](https://img.shields.io/badge/build-passing-green)](https://github.com/Tessellate-Imaging/monk_v1)

## Documentation
- [Introduction](https://clever-noyce-f9d43f.netlify.com/#/introduction)
- [Setup](https://clever-noyce-f9d43f.netlify.com/#/setup/setup)
- [Quick Mode](https://clever-noyce-f9d43f.netlify.com/#/quick_mode/quickmode_pytorch)
- [Update Mode](https://clever-noyce-f9d43f.netlify.com/#/update_mode/update_dataset)
- [Expert Mode](https://clever-noyce-f9d43f.netlify.com/#/expert_mode)
- [Hyper Parameter Analyser](https://clever-noyce-f9d43f.netlify.com/#/hp_finder/model_finder)
- [Compare Experiments](https://clever-noyce-f9d43f.netlify.com/#/compare_experiment)
- [Resume Training](https://clever-noyce-f9d43f.netlify.com/#/resume_training)



## Create an image classification experiment.
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

## Inference

```python
img_name = "./monk/datasets/test/0.jpg";
predictions = ptf.Infer(img_name=img_name, return_raw=True);
print(predictions)
```


## Compare Experiments

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

## TODO-2020 - Backend Support

- [ ] Tensorflow 2.0
- [ ] Chainer
- [ ] Intel VINO toolkit


## TODO-2020 - External Libraries
- [ ] OpenCV
- [ ] Dlib
- [ ] Python PIL
- [ ] TensorRT Acceleration
- [ ] Intel Acceleration
- [ ] Echo AI - for Activation functions


## TODO-2020 - Features
- [ ] Model Visualization
- [ ] Pre-processed data visualization
- [ ] Learned feature visualization
- [ ] NDimensional data input
- [ ] Data input from npy files
- [ ] Data input from hdfs files
- [ ] Support for Dicom data
- [ ] Support for Tiff data
- [ ] Multi-label Image Classification
- [ ] Custom model development


## TODO-2020 - General
- [ ] Incorporate pep coding standards
- [ ] Functional Documentation
- [ ] Tackle Multiple versions of libraries
- [ ] Add unit-testing
- [ ] Contribution guidelines






## Copyright

Copyright 2019 onwards, Tessellate Imaging Private Limited Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. A copy of the License is provided in the LICENSE file in this repository.
