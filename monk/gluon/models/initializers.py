from gluon.models.imports import *
from system.imports import *

#@TraceFunction(trace_args=False, trace_rv=False)
def initialize_network(net, initializer):
    '''
    Get the right initializer for custom network weight initialization

    Args:
        initializer (str): The type of initializer

    Returns:
        gluon mxnet initializer: The type of initializer
    '''
    if(initializer == "xavier_normal"):
        net.initialize(initializer_xavier_normal(), ctx = ctx);
    elif(initializer == "xavier_uniform"):
        net.initialize(initializer_xavier_uniform(), ctx = ctx);
    elif(initializer == "msra"):
        net.initialize(initializer_msra(), ctx = ctx);
    elif(initializer == "normal"):
        net.initialize(initializer_normal(), ctx = ctx);
    elif(initializer == "orthogonal_normal"):
        net.initialize(initializer_orthogonal_normal(), ctx = ctx);
    elif(initializer == "orthogonal_uniform"):
        net.initialize(initializer_orthogonal_uniform(), ctx = ctx);
    elif(initializer == "uniform"):
        net.initialize(initializer_uniform(), ctx = ctx);

    return net;
    


#@TraceFunction(trace_args=False, trace_rv=False)
def initializer_xavier_normal():
    '''
    Get Xavier normal initialization

    Args:
        None

    Returns:
        gluon mxnet initializer: Xavier Normal intializer
    '''
    return initializer.Xavier(rnd_type='gaussian');



#@TraceFunction(trace_args=False, trace_rv=False)
def initializer_xavier_uniform():
    '''
    Get Xavier uniform initialization

    Args:
        None

    Returns:
        gluon mxnet initializer: Xavier uniform intializer
    '''
    return initializer.Xavier(rnd_type='uniform');



#@TraceFunction(trace_args=False, trace_rv=False)
def initializer_msra():
    '''
    Get MSRA initialization

    Args:
        None

    Returns:
        gluon mxnet initializer: MSRA intializer
    '''
    return initializer.MSRAPrelu();



#@TraceFunction(trace_args=False, trace_rv=False)
def initializer_normal():
    '''
    Get normal initialization

    Args:
        None

    Returns:
        gluon mxnet initializer: normal intializer
    '''
    return initializer.Normal();



#@TraceFunction(trace_args=False, trace_rv=False)
def initializer_orthogonal_normal():
    '''
    Get orthogonal normal initialization

    Args:
        None

    Returns:
        gluon mxnet initializer: Orthogonal normal intializer
    '''
    return initializer.Orthogonal(rand_type='normal');




#@TraceFunction(trace_args=False, trace_rv=False)
def initializer_orthogonal_uniform():
    '''
    Get orthogonal uniform initialization

    Args:
        None

    Returns:
        gluon mxnet initializer: Orthogonal uniform intializer
    '''
    return initializer.Orthogonal(rand_type='uniform');




#@TraceFunction(trace_args=False, trace_rv=False)
def initializer_uniform():
    '''
    Get uniform initialization

    Args:
        None

    Returns:
        gluon mxnet initializer: Uniform intializer
    '''
    return initializer.Uniform();