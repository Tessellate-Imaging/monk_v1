from monk.pytorch.models.imports import *
from monk.system.imports import *
from monk.pytorch.models.common import set_parameter_requires_grad


#classifier 6
set1 = ["alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]

#classifier
set2 = ["densenet121", "densenet161", "densenet169", "densenet201"]

#fc
set3 = ["googlenet", "inception_v3", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "wide_resnet101_2", "wide_resnet50_2"]

#classifier 1
set4 = ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3", "mobilenet_v2", "squeezenet1_0", "squeezenet1_1"]

combined_list = set1+set2+set3+set4
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

    if(model_name == "alexnet"):
        finetune_net = torchvision.models.alexnet(pretrained=use_pretrained);
    elif(model_name == "vgg11"):
        finetune_net = torchvision.models.vgg11(pretrained=use_pretrained);
    elif(model_name == "vgg11_bn"):
        finetune_net = torchvision.models.vgg11_bn(pretrained=use_pretrained);
    elif(model_name == "vgg13"):
        finetune_net = torchvision.models.vgg11(pretrained=use_pretrained);
    elif(model_name == "vgg13_bn"):
        finetune_net = torchvision.models.vgg11_bn(pretrained=use_pretrained);
    elif(model_name == "vgg16"):
        finetune_net = torchvision.models.vgg16(pretrained=use_pretrained);
    elif(model_name == "vgg16_bn"):
        finetune_net = torchvision.models.vgg16_bn(pretrained=use_pretrained);
    elif(model_name == "vgg19"):
        finetune_net = torchvision.models.vgg19(pretrained=use_pretrained);
    elif(model_name == "vgg19_bn"):
        finetune_net = torchvision.models.vgg19_bn(pretrained=use_pretrained);
    elif(model_name == "densenet121"):
        finetune_net = torchvision.models.densenet121(pretrained=use_pretrained);
    elif(model_name == "densenet161"):
        finetune_net = torchvision.models.densenet161(pretrained=use_pretrained);
    elif(model_name == "densenet169"):
        finetune_net = torchvision.models.densenet169(pretrained=use_pretrained);
    elif(model_name == "densenet201"):
        finetune_net = torchvision.models.densenet201(pretrained=use_pretrained);
    elif(model_name == "googlenet"):
        finetune_net = torchvision.models.googlenet(pretrained=use_pretrained);
    elif(model_name == "inception_v3"):
        finetune_net = torchvision.models.inception_v3(pretrained=use_pretrained);
    elif(model_name == "resnet18"):
        finetune_net = torchvision.models.resnet18(pretrained=use_pretrained);
    elif(model_name == "resnet34"):
        finetune_net = torchvision.models.resnet34(pretrained=use_pretrained);
    elif(model_name == "resnet50"):
        finetune_net = torchvision.models.resnet50(pretrained=use_pretrained);
    elif(model_name == "resnet101"):
        finetune_net = torchvision.models.resnet101(pretrained=use_pretrained);
    elif(model_name == "resnet152"):
        finetune_net = torchvision.models.resnet152(pretrained=use_pretrained);
    elif(model_name == "resnext50_32x4d"):
        finetune_net = torchvision.models.resnext50_32x4d(pretrained=use_pretrained);
    elif(model_name == "resnext101_32x8d"):
        finetune_net = torchvision.models.resnext101_32x8d(pretrained=use_pretrained);
    elif(model_name == "shufflenet_v2_x0_5"):
        finetune_net = torchvision.models.shufflenet_v2_x0_5(pretrained=use_pretrained);
    elif(model_name == "shufflenet_v2_x1_0"):
        finetune_net = torchvision.models.shufflenet_v2_x1_0(pretrained=use_pretrained);
    elif(model_name == "shufflenet_v2_x1_5"):
        if(use_pretrained):
            msg = "Pretrained model Unavailable for {}.\n".format(model_name);
            msg += "Using xavier initialization";
            ConstraintWarning(msg);
        finetune_net = torchvision.models.shufflenet_v2_x1_5(pretrained=False);
    elif(model_name == "shufflenet_v2_x2_0"):
        if(use_pretrained):
            msg = "Pretrained model Unavailable for {}.\n".format(model_name);
            msg += "Using xavier initialization";
            ConstraintWarning(msg);
        finetune_net = torchvision.models.shufflenet_v2_x2_0(pretrained=False);
    elif(model_name == "wide_resnet101_2"):
        finetune_net = torchvision.models.wide_resnet101_2(pretrained=use_pretrained);
    elif(model_name == "wide_resnet50_2"):
        finetune_net = torchvision.models.wide_resnet50_2(pretrained=use_pretrained);
    elif(model_name == "mnasnet0_5"):
        finetune_net = torchvision.models.mnasnet0_5(pretrained=use_pretrained);
    elif(model_name == "mnasnet0_75"):
        if(use_pretrained):
            msg = "Pretrained model Unavailable for {}.\n".format(model_name);
            msg += "Using xavier initialization";
            ConstraintWarning(msg);
        finetune_net = torchvision.models.mnasnet0_75(pretrained=False);
    elif(model_name == "mnasnet1_0"):
        finetune_net = torchvision.models.mnasnet1_0(pretrained=use_pretrained);
    elif(model_name == "mnasnet1_3"):
        if(use_pretrained):
            msg = "Pretrained model Unavailable for {}.\n".format(model_name);
            msg += "Using xavier initialization";
            ConstraintWarning(msg);
        finetune_net = torchvision.models.mnasnet1_3(pretrained=False);
    elif(model_name == "mobilenet_v2"):
        finetune_net = torchvision.models.mobilenet_v2(pretrained=use_pretrained);
    elif(model_name == "squeezenet1_0"):
        finetune_net = torchvision.models.squeezenet1_0(pretrained=use_pretrained);
    elif(model_name == "squeezenet1_1"):
        finetune_net = torchvision.models.squeezenet1_1(pretrained=use_pretrained);


    finetune_net = set_parameter_requires_grad(finetune_net, freeze_base_network);

    return finetune_net, model_name;

