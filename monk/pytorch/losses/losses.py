from pytorch.losses.imports import *
from system.imports import *


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], size_average=[list, type(np.array([1, 2, 3])), float, type(None)], 
    ignore_index=int, reduction=str, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def softmax_crossentropy(system_dict, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
    system_dict["local"]["criterion"] = "softmaxcrossentropy";
    system_dict["hyper-parameters"]["loss"]["name"] = "softmaxcrossentropy";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["size_average"] = size_average;
    system_dict["hyper-parameters"]["loss"]["params"]["ignore_index"] = ignore_index;
    system_dict["hyper-parameters"]["loss"]["params"]["reduce"] = None;
    system_dict["hyper-parameters"]["loss"]["params"]["reduction"] = reduction;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;



@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], size_average=[list, type(np.array([1, 2, 3])), float, type(None)], 
    ignore_index=int, reduction=str, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def nll(system_dict, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    system_dict["local"]["criterion"] = "nll";
    system_dict["hyper-parameters"]["loss"]["name"] = "nll";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["size_average"] = size_average;
    system_dict["hyper-parameters"]["loss"]["params"]["ignore_index"] = ignore_index;
    system_dict["hyper-parameters"]["loss"]["params"]["reduce"] = reduce;
    system_dict["hyper-parameters"]["loss"]["params"]["reduction"] = reduction;
    return system_dict;




@accepts(dict, log_input=bool, full=bool, size_average=[list, type(np.array([1, 2, 3])), float, type(None)], epsilon=[int, float], 
    reduce=type(None), reduction=str, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def poisson_nll(system_dict, log_input=True, full=False, size_average=None, epsilon=1e-08, reduce=None, reduction='mean'):
    system_dict["local"]["criterion"] = "poissonnll";
    system_dict["hyper-parameters"]["loss"]["name"] = "poissonnll";
    system_dict["hyper-parameters"]["loss"]["params"]["log_input"] = log_input;
    system_dict["hyper-parameters"]["loss"]["params"]["full"] = full;
    system_dict["hyper-parameters"]["loss"]["params"]["size_average"] = size_average;
    system_dict["hyper-parameters"]["loss"]["params"]["reduce"] = reduce;
    system_dict["hyper-parameters"]["loss"]["params"]["reduction"] = reduction;
    system_dict["hyper-parameters"]["loss"]["params"]["eps"] = epsilon;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;



@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], size_average=[list, type(np.array([1, 2, 3])), float, type(None)], 
    reduce=type(None), reduction=str, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def binary_crossentropy(system_dict, weight=None, size_average=None, reduce=None, reduction='mean'):
    system_dict["local"]["criterion"] = "binarycrossentropy";
    system_dict["hyper-parameters"]["loss"]["name"] = "binarycrossentropy";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["size_average"] = size_average;
    system_dict["hyper-parameters"]["loss"]["params"]["reduce"] = reduce;
    system_dict["hyper-parameters"]["loss"]["params"]["reduction"] = reduction;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;



@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], size_average=[list, type(np.array([1, 2, 3])), float, type(None)], 
    reduce=type(None), reduction=str, pos_weight=[list, type(np.array([1, 2, 3])), float, type(None)], post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def binary_crossentropy_with_logits(system_dict, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
    system_dict["local"]["criterion"] = "binarycrossentropywithlogits";
    system_dict["hyper-parameters"]["loss"]["name"] = "binarycrossentropywithlogits";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["size_average"] = size_average;
    system_dict["hyper-parameters"]["loss"]["params"]["reduce"] = reduce;
    system_dict["hyper-parameters"]["loss"]["params"]["reduction"] = reduction;
    system_dict["hyper-parameters"]["loss"]["params"]["pos_weight"] = pos_weight;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;

