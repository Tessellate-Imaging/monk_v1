from monk.gluon.models.imports import *
from monk.system.imports import *
from monk.gluon.models.common import set_parameter_requires_grad



set1 = ["alexnet", "darknet53", "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201", "InceptionV3", "MobileNet1.0", "MobileNet0.75", 
            "MobileNet0.25", "MobileNet0.5", "ResNet18_v1", "ResNet34_v1", "ResNet50_v1", "ResNet101_v1", "ResNet152_v1", "ResNext50_32x4d", 
            "ResNext101_32x4d", "ResNext101_64x4d", "SE_ResNext50_32x4d", "SE_ResNext101_32x4d", "SE_ResNext101_64x4d", "SENet_154", 
            "VGG11", "VGG13", "VGG16", "VGG19", "VGG11_bn", "VGG13_bn", "VGG16_bn", "VGG19_bn", "ResNet18_v2", "ResNet34_v2", 
            "ResNet50_v2", "ResNet101_v2", "ResNet152_v2"];
set2 = ["MobileNetV2_1.0", "MobileNetV2_0.75", "MobileNetV2_0.5", "MobileNetV2_0.25", "SqueezeNet1.0", "SqueezeNet1.1", "MobileNetV3_Large", "MobileNetV3_Small"];
set3 = ["ResNet18_v1b", "ResNet34_v1b", "ResNet50_v1b", "ResNet50_v1b_gn", "ResNet101_v1b", "ResNet152_v1b", "ResNet50_v1c", 
            "ResNet101_v1c", "ResNet152_v1c", "ResNet50_v1d", "ResNet101_v1d", "ResNet152_v1d", "ResNet18_v1d", "ResNet34_v1d", 
            "ResNet50_v1d", "ResNet101_v1d", "ResNet152_v1d", "resnet18_v1b_0.89", "resnet50_v1d_0.86", "resnet50_v1d_0.48", 
            "resnet50_v1d_0.37", "resnet50_v1d_0.11", "resnet101_v1d_0.76", "resnet101_v1d_0.73", "Xception"];
combined_list = set1+set2+set3
combined_list_lower = list(map(str.lower, combined_list))


@accepts(str, bool, int, bool, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def get_base_model(model_name, use_pretrained, num_classes, freeze_base_network):
    '''
    Get base network for transfer learning based on parameters selected

    Args:
        model_name (str): Select from available models. Check via List_Models() function
        freeze_base_network (bool): If set as True, then base network's weights are freezed (cannot be trained)
        use_gpu (bool): If set as True, uses GPU
        use_pretrained (bool): If set as True, use weights trained on imagenet and coco like dataset
                                Else, use randomly initialized weights.

    Returns:
        neural network: Base network
        str: Name of the model
    '''
    if(model_name not in combined_list_lower):
        print("Model name: {} not found".format(model_name));
    else:
        index = combined_list_lower.index(model_name);
        model_name = combined_list[index];
        finetune_net = get_model(model_name, pretrained=use_pretrained);    
    finetune_net = set_parameter_requires_grad(finetune_net, freeze_base_network);

    return finetune_net, model_name;








