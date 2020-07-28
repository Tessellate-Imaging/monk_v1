from monk.tf_keras_1.models.imports import *
from monk.system.imports import *
from monk.tf_keras_1.models.common import set_parameter_requires_grad


#classifier 6
set1 = ["mobilenet", "densenet121", "densenet169", "densenet201", "inception_v3", 
        "inception_resnet_v3", "mobilenet_v2", "nasnet_mobile", "nasnet_large", "resnet50",
        "resnet101", "resnet152", "resnet50_v2", "resnet101_v2", "resnet152_v2", "vgg16",
        "vgg19", "xception"];

combined_list = set1
combined_list_lower = list(map(str.lower, combined_list))


@accepts(str, bool, int, bool, int, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def get_base_model(model_name, use_pretrained, num_classes, freeze_base_network, input_size):
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

        if(use_pretrained):
            weights="imagenet";
        else:
            weights=None;


        if(model_name == "mobilenet"):
            from keras.applications import MobileNet as keras_model
        elif(model_name == "densenet121"):
            from keras.applications import DenseNet121 as keras_model
        elif(model_name == "densenet169"):
            from keras.applications import DenseNet169 as keras_model
        elif(model_name == "densenet201"):
            from keras.applications import DenseNet201 as keras_model
        elif(model_name == "inception_v3"):
            from keras.applications import InceptionV3 as keras_model
        elif(model_name == "inception_resnet_v3"):
            from keras.applications import InceptionResNetV2 as keras_model
        elif(model_name == "mobilenet_v2"):
            from keras.applications import MobileNetV2 as keras_model
        elif(model_name == "nasnet_mobile"):
            from keras.applications import NASNetMobile as keras_model
        elif(model_name == "nasnet_large"):
            from keras.applications import NASNetLarge as keras_model
        elif(model_name == "resnet50"):
            from keras.applications import ResNet50 as keras_model
        elif(model_name == "resnet101"):
            from keras.applications import ResNet101 as keras_model
        elif(model_name == "resnet152"):
            from keras.applications import ResNet152 as keras_model
        elif(model_name == "resnet50_v2"):
            from keras.applications import ResNet50V2 as keras_model
        elif(model_name == "resnet101_v2"):
            from keras.applications import ResNet101V2 as keras_model
        elif(model_name == "resnet152_v2"):
            from keras.applications import ResNet152V2 as keras_model
        elif(model_name == "vgg16"):
            from keras.applications import VGG16 as keras_model
        elif(model_name == "vgg19"):
            from keras.applications import VGG19 as keras_model
        elif(model_name == "xception"):
            from keras.applications import Xception as keras_model


        finetune_net = keras_model(weights=weights, include_top=False, input_shape=(input_size, input_size, 3));

        finetune_net = set_parameter_requires_grad(finetune_net, freeze_base_network);

        return finetune_net, model_name;