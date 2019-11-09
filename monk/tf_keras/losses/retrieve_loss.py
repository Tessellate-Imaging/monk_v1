from tf_keras.losses.imports import *
from system.imports import *


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_loss(system_dict):
    system_dict["local"]["criterion"] = system_dict["hyper-parameters"]["loss"]["name"];
    name = system_dict["local"]["criterion"];


    if(name == "categoricalcrossentropy"):
        system_dict["local"]["criterion"] = krlo.categorical_crossentropy;

    elif(name == "sparsecategoricalcrossentropy"):
        system_dict["local"]["criterion"] = krlo.sparse_categorical_crossentropy;

    elif(name == "categoricalhinge"):
        system_dict["local"]["criterion"] = krlo.categorical_hinge;

    elif(name == "binarycrossentropy"):
        system_dict["local"]["criterion"] = krlo.binary_crossentropy;

    return system_dict;