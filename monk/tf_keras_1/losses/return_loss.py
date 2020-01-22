from tf_keras_1.losses.imports import *
from system.imports import *


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def load_loss(system_dict):
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



    '''
    if(name == "categoricalcrossentropy"):
        system_dict["local"]["criterion"] = krlo.categorical_crossentropy;

    elif(name == "sparsecategoricalcrossentropy"):
        system_dict["local"]["criterion"] = krlo.sparse_categorical_crossentropy;

    elif(name == "categoricalhinge"):
        system_dict["local"]["criterion"] = krlo.categorical_hinge;

    elif(name == "binarycrossentropy"):
        system_dict["local"]["criterion"] = krlo.binary_crossentropy;
    '''


    return system_dict;