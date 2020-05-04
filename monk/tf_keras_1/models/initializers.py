from tf_keras_1.models.imports import *
from system.imports import *


#@TraceFunction(trace_args=False, trace_rv=False)
def get_initializer(initializer):
    '''
    Get the right initializer for custom network weight initialization

    Args:
        initializer (str): The type of initializer

    Returns:
        str: The type of initializer
    '''

    if(initializer == "xavier_normal"):
        return "glorot_normal";
    elif(initializer == "xavier_uniform"):
        return "glorot_uniform";
    elif(initializer == "random_uniform"):
        return "random_uniform";
    elif(initializer == "random_normal"):
        return "random_normal";
    elif(initializer == "lecun_uniform"):
        return "lecun_uniform";
    elif(initializer == "lecun_normal"):
        return "lecun_normal";
    elif(initializer == "he_normal"):
        return "he_normal";
    elif(initializer == "he_uniform"):
        return "he_uniform";
    elif(initializer == "truncated_normal"):
        return "truncated_normal";
    elif(initializer == "orthogonal"):
        return "orthogonal";
    elif(initializer == "variance_scaling"):
        return "VarianceScaling";