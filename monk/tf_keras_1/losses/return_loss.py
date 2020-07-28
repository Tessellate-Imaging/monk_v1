from monk.tf_keras_1.losses.imports import *
from monk.system.imports import *


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def load_loss(system_dict):
    '''
    Load loss function in native library

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    name = system_dict["local"]["criterion"];

    if(name == "l1"):
        system_dict["local"]["criterion"] = krlo.mean_absolute_error

    elif(name == "l2"):
        system_dict["local"]["criterion"] = krlo.mean_squared_error

    elif(name == "crossentropy"):
        system_dict["local"]["criterion"] = krlo.categorical_crossentropy;

    elif(name == "binarycrossentropy"):
        system_dict["local"]["criterion"] = krlo.binary_crossentropy;

    elif(name == "kldiv"):
        system_dict["local"]["criterion"] = krlo.kullback_leibler_divergence;

    elif(name == "hinge"):
        system_dict["local"]["criterion"] = krlo.hinge;  

    elif(name == "squaredhinge"):
        system_dict["local"]["criterion"] = krlo.squared_hinge;

    return system_dict;