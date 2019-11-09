from gluon.optimizers.imports import *
from system.imports import *


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_optimizer(system_dict):
    system_dict["local"]["optimizer"] = system_dict["hyper-parameters"]["optimizer"]["name"];
    return system_dict;