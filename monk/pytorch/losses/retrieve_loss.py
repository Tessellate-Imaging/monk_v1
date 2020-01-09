from pytorch.losses.imports import *
from system.imports import *


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_loss(system_dict):
    system_dict["local"]["criterion"] = system_dict["hyper-parameters"]["loss"]["name"];
    name = system_dict["local"]["criterion"];


    if(name == "l1"):
        system_dict["local"]["criterion"] = torch.nn.L1Loss(
            reduction='mean');

    elif(name == "l2"):
        system_dict["local"]["criterion"] = torch.nn.MSELoss(
            reduction='mean');

    elif(name == "softmaxcrossentropy"):
        system_dict["local"]["criterion"] = torch.nn.CrossEntropyLoss(
            weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"], 
            reduction='mean');

    elif(name == "crossentropy"):
        system_dict["local"]["criterion"] = torch.nn.NLLLoss(
            weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"], 
            reduction='mean');

    elif(name == "sigmoidbinarycrossentropy"):
        system_dict["local"]["criterion"] = torch.nn.BCEWithLogitsLoss(
            weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"], 
            reduction='mean');

    elif(name == "binarycrossentropy"):
        system_dict["local"]["criterion"] = torch.nn.BCELoss(
            weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"], 
            reduction='mean');

    elif(name == "kldiv"):
        system_dict["local"]["criterion"] = torch.nn.KLDivLoss(
            reduction='mean');

    elif(name == "poissonnll"):
        system_dict["local"]["criterion"] = torch.nn.PoissonNLLLoss(
            log_input=system_dict["hyper-parameters"]["loss"]["params"]["log_pre_applied"],
            reduction='mean');

    elif(name == "huber"):
        system_dict["local"]["criterion"] = torch.nn.SmoothL1Loss(
            reduction='mean');

    elif(name == "hinge"):
        system_dict["local"]["criterion"] = torch.nn.HingeEmbeddingLoss(
            margin=system_dict["hyper-parameters"]["loss"]["params"]["margin"],
            reduction='mean');

    elif(name == "squaredhinge"):
        system_dict["local"]["criterion"] = torch.nn.SoftMarginLoss(
            reduction='mean');

    elif(name == "multimargin"):
        system_dict["local"]["criterion"] = torch.nn.MultiMarginLoss(
            p=1,
            margin=system_dict["hyper-parameters"]["loss"]["params"]["margin"],
            reduction='mean');

    elif(name == "squaredmultimargin"):
        system_dict["local"]["criterion"] = torch.nn.MultiMarginLoss(
            p=2,
            margin=system_dict["hyper-parameters"]["loss"]["params"]["margin"],
            reduction='mean');

    elif(name == "multilabelmargin"):
        system_dict["local"]["criterion"] = torch.nn.MultiLabelMarginLoss(
            reduction='mean');

    elif(name == "multilabelsoftmargin"):
        system_dict["local"]["criterion"] = torch.nn.MultiLabelSoftMarginLoss(
            reduction='mean');




    '''
    if(name == "softmaxcrossentropy"):
        system_dict["local"]["criterion"] = torch.nn.CrossEntropyLoss(
            weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"], 
            size_average=system_dict["hyper-parameters"]["loss"]["params"]["size_average"], 
            ignore_index=system_dict["hyper-parameters"]["loss"]["params"]["ignore_index"], 
            reduce=system_dict["hyper-parameters"]["loss"]["params"]["reduce"], 
            reduction=system_dict["hyper-parameters"]["loss"]["params"]["reduction"]);

    elif(name == "nll"):
        system_dict["local"]["criterion"] = torch.nn.NLLLoss(
            weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"], 
            size_average=system_dict["hyper-parameters"]["loss"]["params"]["size_average"], 
            ignore_index=system_dict["hyper-parameters"]["loss"]["params"]["ignore_index"], 
            reduce=system_dict["hyper-parameters"]["loss"]["params"]["reduce"], 
            reduction=system_dict["hyper-parameters"]["loss"]["params"]["reduction"]);

    elif(name == "poissonnll"):
        system_dict["local"]["criterion"] = torch.nn.PoissonNLLLoss(
            log_input=system_dict["hyper-parameters"]["loss"]["params"]["log_input"], 
            full=system_dict["hyper-parameters"]["loss"]["params"]["log_input"], 
            size_average=system_dict["hyper-parameters"]["loss"]["params"]["log_input"], 
            eps=system_dict["hyper-parameters"]["loss"]["params"]["log_input"], 
            reduce=system_dict["hyper-parameters"]["loss"]["params"]["reduce"], 
            reduction=system_dict["hyper-parameters"]["loss"]["params"]["reduction"]);

    elif(name == "binarycrossentropy"):
        system_dict["local"]["criterion"] = torch.nn.BCELoss(
            weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"], 
            size_average=system_dict["hyper-parameters"]["loss"]["params"]["size_average"], 
            reduce=system_dict["hyper-parameters"]["loss"]["params"]["reduce"], 
            reduction=system_dict["hyper-parameters"]["loss"]["params"]["reduction"]);

    elif(name == "binarycrossentropywithlogits"):
        system_dict["local"]["criterion"] = torch.nn.BCEWithLogitsLoss(
            weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"], 
            size_average=system_dict["hyper-parameters"]["loss"]["params"]["size_average"], 
            reduce=system_dict["hyper-parameters"]["loss"]["params"]["reduce"], 
            reduction=system_dict["hyper-parameters"]["loss"]["params"]["reduction"], 
            pos_weight=system_dict["hyper-parameters"]["loss"]["params"]["pos_weight"]);
    '''

    return system_dict;