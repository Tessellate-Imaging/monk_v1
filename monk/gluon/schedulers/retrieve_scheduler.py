from monk.gluon.schedulers.imports import *
from monk.system.imports import *

@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_scheduler(system_dict):
    '''
    Retrieve schedulers in validation, testing, copy-from and resume states

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    if(system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] == "fixed"):
        system_dict["local"]["learning_rate_scheduler"] = None;
    else:
        system_dict["local"]["learning_rate_scheduler"] = system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"];

    return system_dict;