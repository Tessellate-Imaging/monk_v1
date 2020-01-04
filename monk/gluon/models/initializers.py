from gluon.models.imports import *
from system.imports import *

@TraceFunction(trace_args=False, trace_rv=False)
def initialize_network(net, initializer):
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
    


@TraceFunction(trace_args=False, trace_rv=False)
def initializer_xavier_normal():
    return initializer.Xavier(rnd_type='gaussian');

@TraceFunction(trace_args=False, trace_rv=False)
def initializer_xavier_uniform():
    return initializer.Xavier(rnd_type='uniform');

@TraceFunction(trace_args=False, trace_rv=False)
def initializer_msra():
    return initializer.MSRAPrelu();

@TraceFunction(trace_args=False, trace_rv=False)
def initializer_normal():
    return initializer.Normal();

@TraceFunction(trace_args=False, trace_rv=False)
def initializer_orthogonal_normal():
    return initializer.Orthogonal(rand_type='normal');

@TraceFunction(trace_args=False, trace_rv=False)
def initializer_orthogonal_uniform():
    return initializer.Orthogonal(rand_type='uniform');

@TraceFunction(trace_args=False, trace_rv=False)
def initializer_uniform():
    return initializer.Uniform();