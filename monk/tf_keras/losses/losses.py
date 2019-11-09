from tf_keras.losses.imports import *
from system.imports import *



@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], size_average=[list, type(np.array([1, 2, 3])), float, type(None)], 
    ignore_index=int, reduction=str, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def categorical_crossentropy(system_dict, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
    system_dict["local"]["criterion"] = "categoricalcrossentropy";
    system_dict["hyper-parameters"]["loss"]["name"] = "categoricalcrossentropy";
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
def sparse_categorical_crossentropy(system_dict, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
    system_dict["local"]["criterion"] = "sparsecategoricalcrossentropy";
    system_dict["hyper-parameters"]["loss"]["name"] = "sparsecategoricalcrossentropy";
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
def categorical_hinge(system_dict, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
    system_dict["local"]["criterion"] = "categoricalhinge";
    system_dict["hyper-parameters"]["loss"]["name"] = "categoricalhinge";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["size_average"] = size_average;
    system_dict["hyper-parameters"]["loss"]["params"]["ignore_index"] = ignore_index;
    system_dict["hyper-parameters"]["loss"]["params"]["reduce"] = None;
    system_dict["hyper-parameters"]["loss"]["params"]["reduction"] = reduction;
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