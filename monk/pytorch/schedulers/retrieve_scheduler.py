from pytorch.schedulers.imports import *
from system.imports import *

@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_scheduler(system_dict):
    if(system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] == "fixed"):
        system_dict["local"]["learning_rate_scheduler"] = None;
    else:
        system_dict["local"]["learning_rate_scheduler"] = system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"];

    return system_dict;