# Table of Contents
    - 1_getting_started_roadmap: For the first timers using Monk
    - 2_transfer_learning_roadmap: Learn comprehensively about transfer learning using monk
    - 3_image_processing_deep_learning_roadmap: Learn roots of image processing and deep learning using monk
    - 4_image_classification_zoo: A huge list of image classification examples with pre-trained models




# List of all available functions and associated tutorials

 - [Available Backend frameworks](#1) 
 - [Available Transfer Learning Models](#2)
 - [Available Layers](#3)
 - [Available Activation Functions](#4)
 - [Available Optimizers](#5)
 - [Available Loss functions](#6)



<br />
<br />
<br />

<a id="1"></a>
## Available backend frameworks
  
    a) Mxnet Gluon - version 1.5.1
    b) Pytorch     - version 1.2.0
    c) Keras       - version 2.2.5 (tf - 1.12.0)
    

<br />
<br />
<br />

<a id="2"></a>
## Available Transfer Learning Models

| monk name           | Original Name in Keras | Original Name in Pytorch | Original Name in Pytorch |
|---------------------|------------------------|--------------------------|--------------------------|
| alexnet             | -                      | alexnet                  | AlexNet                  |
| darknet             | -                      | -                        | Darnet53                 |
| densenet121         | DenseNet121            | densenet121              | DenseNet121              |
| densenet161         | -                      | densenet161              | DenseNet161              |
| densenet169         | DenseNet169            | densenet169              | DenseNet169              |
| densenet201         | DenseNet201            | densenet201              | DenseNet201              |
| googlenet           | -                      | googlenet                | -                        |
| inception_v3        | InceptionV3            | inception_v3             | InceptionV3              |
| inception_resnet_v2 | InceptionResNetV2      | -                        | -                        |
| mnasnet0_5          | -                      | mnasnet0_5               | -                        |
| mnasnet0_75         | -                      | mnasnet0_75              | -                        |
| mnasnet1_0          | -                      | mnasnet1_0               | -                        |
| mnasnet1_3          | -                      | mnasnet1_3               | -                        |
| nasnet_mobile       | NASNetMobile           | -                        | -                        |
| nasnet_large        | NASNetLarge            | -                        | -                        |
| mobilenet           | MobileNet              | -                        | MobileNet1.0             |
| mobilenet1.0_int8   | -                      | -                        | MobileNet1.0_int8        |
| mobilenet0.75       | -                      | -                        | MobileNet0.75            |
| mobilenet0.5        | -                      | -                        | MobileNet0.5             |
| mobilenet0.25       | -                      | -                        | MobileNet0.25            |
| mobilenetv2         | MobileNetV2            | mobilenet_v2             | MobileNetV2_1.0          |
| mobilenetv2_0.75    | -                      | -                        | MobileNetV2_0.75         |
| mobilenetv2_0.5     | -                      | -                        | MobileNetV2_0.5          |
| mobilenetv2_0.25    | -                      | -                        | MobileNetV2_0.25         |
| mobilenetv3_large   | -                      | -                        | MobileNetV3_Large        |
| mobilenetv3_smalle  | -                      | -                        | MobileNetV3_Small        |
| resnet18_v1         | -                      | resnet18                 | ResNet18_v1              |
| resnet34_v1         | -                      | resnet34                 | ResNet34_v1              |
| resnet50_v1         | ResNet50               | resnet50                 | ResNet50_v1              |
| resnet101_v1        | ResNet101              | resnet101                | ResNet101_v1             |
| resnet152_v1        | ResNet152              | resnet152                | ResNet152_v1             |
| resnet18_v2         |                        |                          | ResNet18_v2              |
| resnet34_v2         |                        |                          | ResNet34_v2              |
| resnet50_v2         | ResNet50V2             | -                        | ResNet50_v2              |
| resnet101_v2        | ResNet101V2            | -                        | ResNet101_v2             |
| resnet152_v2        | ResNet152V2            | -                        | ResNet152_v2             |
| resnext50_32x4d     | -                      | resnext50_32x4d          | ResNext50_32x4d          |
| resnext101_32x8d    | -                      | resnext101_32x8d         | ResNext101_32x4d         |
| resnext101_64x4d    | -                      | -                        | ResNext101_64x4d         |
| se_resnext50_32x4d  | -                      | -                        | SE_ResNext50_32x4d       |
| se_resnext101_32x4d | -                      | -                        | SE_ResNext101_32x4d      |
| se_resnext101_64x4d | -                      | -                        | SE_ResNext101_64x4d      |
| shufflenet_v2_x0_5  | -                      | shufflenet_v2_x0_5       | -                        |
| shufflenet_v2_x1_0  | -                      | shufflenet_v2_x1_0       | -                        |
| shufflenet_v2_x1_5  | -                      | shufflenet_v2_x1_5       | -                        |
| shufflenet_v2_x2_0  | -                      | shufflenet_v2_x2_0       | -                        |
| squeezenet1_0       | -                      | squeezenet1_0            | SqueezeNet1.0            |
| squeezenet1_1       | -                      | squeezenet1_1            | SqueezeNet1.1            |
| senet_154           | -                      | -                        | SENet_154                |
| vgg11               | -                      | vgg11                    | VGG11                    |
| vgg11_bn            | -                      | vgg11_bn                 | VGG11_bn                 |
| vgg13               | -                      | vgg13                    | VGG13                    |
| vgg13_bn            | -                      | vgg13_bn                 | VGG13_bn                 |
| vgg16               | VGG16                  | vgg16                    | VGG16                    |
| vgg16_bn            | -                      | vgg16_bn                 | VGG16_bn                 |
| vgg19               | VGG19                  | vgg19                    | VGG19                    |
| vgg19_bn            | -                      | vgg19_bn                 | VGG19_bn                 |
| wide_resnet50_2     | -                      | wide_resnet50_2          | -                        |
| wide_resnet101_2    | -                      | wide_resnet101_2         | -                        |
| xception            | Xception               | -                        | Xception                 |



<br />
<br />
<br />

<a id="3"></a>
## Available Custom Network Layers

| Name in Monk             | Name in Keras backend  | Name in pytorch backend           | Name in mxnet backed |
|--------------------------|------------------------|-----------------------------------|----------------------|
| fully_connected          | Dense                  | Linear                            | Dense                |
| Dropout                  | Dropout                | Dropout                           | Dropout              |
| Flatten                  | Flatten                | Flatten                           | Flatten              |
| convolution1d            | Conv1D                 | Conv1d                            | Conv1D               |
| convolution              | Conv2D                 | Conv2d                            | Conv2D               |
| convolution3d            | Conv3D                 | Conv3d                            | Conv3D               |
| transposed_convolution1d | -                      | ConvTranspose1d                   | Conv1DTranspose      |
| transposed_convolution   | Conv2DTranspose        | ConvTranspose2d                   | Conv2DTranspose      |
| transposed_convolution3d | Conv3DTranspose        | ConvTranspose3d                   | Conv3DTranspose      |
| max_pooling1d            | MaxPooling1D           | MaxPool1d                         | MaxPool1D            |
| max_pooling              | MaxPooling2D           | MaxPool2d                         | MaxPool2D            |
| max_pooling3d            | MaxPooling3D           | MaxPool3d                         | MaxPool3D            |
| average_pooling1d        | AveragePooling1D       | AvgPool1d                         | AvgPool1D            |
| average_pooling          | AveragePooling2D       | AvgPool2d                         | AvgPool2D            |
| average_pooling3d        | AveragePooling3D       | AvgPool3d                         | AvgPool3D            |
| global_max_pooling1d     | GlobalMaxPooling1D     | AdaptiveMaxPool1d (With size = 1) | GlobalMaxPool1D      |
| global_max_pooling       | GlobalMaxPooling2D     | AdaptiveMaxPool2d (With size = 1) | GlobalMaxPool2D      |
| global_max_pooling3d     | GlobalMaxPooling3D     | AdaptiveMaxPool3d (With size = 1) | GlobalMaxPool3D      |
| global_average_pooling1d | GlobalAveragePooling1D | AdaptiveAvgPool1d (With size = 1) | GlobalAvgPool1D      |
| global_average_pooling   | GlobalAveragePooling2D | AdaptiveAvgPool2d (With size = 1) | GlobalAvgPool2D      |
| global_average_pooling3d | GlobalAveragePooling3D | AdaptiveAvgPool3d (With size = 1) | GlobalAvgPool3D      |
| add                      | Add                    | Add                               | Add                  |
| concatenate              | Concatenate            | Concatenate                       | Concatenate          |
| batchnorm                | BatchNormalization     | BatchNorm1d                       | BatchNorm            |
| batchnorm                | -                      | BatchNorm2d                       | -                    |
| batchnorm                | -                      | BatchNorm3d                       | -                    |
| instancenorm             | -                      | InstanceNorm1d                    | InstanceNorm         |
| instancenorm             | -                      | InstanceNorm2d                    | -                    |
| instancenorm             | -                      | InstanceNorm3d                    | -                    |
| layernorm                | -                      | LayerNorm                         | LayerNorm            |
| identity                 | activation.linear      | Identity                          | Identity             |



<br />
<br />
<br />

<a id="4"></a>
## Available Custom Network Activation Functions

| Name in Monk     | Original name in Keras backend | Original name in pytorch backend | Original name in mxnet backend |
|------------------|--------------------------------|----------------------------------|--------------------------------|
| relu             | relu                           | ReLU                             | Activation('relu')             |
| sigmoid          | sigmoid                        | Sigmoid                          | Activation('sigmoid')          |
| Tanh Shrink      | tanh                           | TanH                             | Activation('tanh')             |
| softplus         | softplus                       | Softplus                         | Activation('softrelu')         |
| softsign         | softsign                       | Softsign                         | Activation('softsign')         |
| elu              | elu                            | ELU                              | ELU                            |
| gelu             | -                              | -                                | GELU                           |
| prelu            | PReLU                          | PReLU                            | PReLU                          |
| selu             | selu                           | SELU                             | SELU                           |
| swish            | -                              | -                                | Swish                          |
| leakyrelu        | LeakyReLU                      | LeakyReLU                        | LeakyReLU                      |
| hardshrink       | -                              | HardShrink                       | -                              |
| hardtanh         | -                              | HardTanh                         | -                              |
| logsigmoid       | -                              | LogSigmoid                       | -                              |
| relu6            | -                              | ReLU6                            | -                              |
| rrelu            | -                              | RReLU                            | -                              |
| celu             | -                              | CELU                             | -                              |
| softshrink       | -                              | Softshrink                       | -                              |
| tanhshrink       | -                              | Tanhshrink                       | -                              |
| threshold        | -                              | Threshold                        | -                              |
| softmin          | -                              | Softmin                          | -                              |
| softmax          | -                              | Softmax                          | -                              |
| logsoftmax       | -                              | LogSoftmax                       | -                              |
| hardsigmoid      | hard_sigmoid                   | -                                | -                              |
| thresholded_relu | ThresholdedReLU                | -                                | -                              |



<br />
<br />
<br />

<a id="4"></a>
## Available Optimizers
