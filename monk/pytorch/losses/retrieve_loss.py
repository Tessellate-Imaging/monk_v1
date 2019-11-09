from pytorch.losses.imports import *
from system.imports import *


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_loss(system_dict):
    system_dict["local"]["criterion"] = system_dict["hyper-parameters"]["loss"]["name"];
    name = system_dict["local"]["criterion"];

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


    return system_dict;