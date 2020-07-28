from monk.tf_keras_1.optimizers.imports import *
from monk.system.imports import *


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_optimizer(system_dict):
    '''
    Retrieve Optimizers in validation, testing, copy-from and resume modes

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = system_dict["hyper-parameters"]["optimizer"]["name"];
    return system_dict;