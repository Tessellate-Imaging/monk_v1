from tf_keras_1.transforms.imports import *
from tf_keras_1.transforms.common import set_transforms
from system.imports import *


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_trainval_transforms(system_dict):
    '''
    Retrieve training and validdation transforms in copy-from, and resume states

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    set_phases = ["train", "val"];
    system_dict = set_transforms(system_dict, set_phases);
    return system_dict;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_test_transforms(system_dict):
    '''
    Retrieve testing transforms in copy-from, and resume states

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    set_phases = ["test"];
    system_dict= set_transforms(system_dict, set_phases);
    return system_dict;