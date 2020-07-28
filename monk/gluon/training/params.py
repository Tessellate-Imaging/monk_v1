from monk.gluon.training.imports import *
from monk.system.imports import *


@accepts(int, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_num_epochs(num_epochs, system_dict):
    '''
    Set number of training epochs

    Args:
        num_epochs (int): Number of epochs to train the network
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    system_dict["hyper-parameters"]["num_epochs"] = num_epochs;
    return system_dict;



@accepts(bool, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_display_progress_realtime(value, system_dict):
    '''
    Set verbosity levels for iterations

    Args:
        value (bool): If True, displays progress for every iteration in the epoch
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    system_dict["training"]["settings"]["display_progress_realtime"] = value;
    return system_dict;


@accepts(bool, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_display_progress(value, system_dict):
    '''
    Set all training params for epochs

    Args:
        value (bool): If True, displays summary post every epoch
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    system_dict["training"]["settings"]["display_progress"] = value;
    return system_dict;


@accepts(bool, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_save_intermediate_models(value, system_dict):
    '''
    Set whether to save models post every epoch or not

    Args:
        value (bool): If True, saves model weight post every epoch
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    system_dict["training"]["settings"]["save_intermediate_models"] = value;
    return system_dict;


@accepts(bool, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_save_training_logs(value, system_dict):
    '''
    Set whether to save training logs or not

    Args:
        value (bool): If True, saves all training and validation metrics. Required for comparison.
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    system_dict["training"]["settings"]["save_training_logs"] = value;
    return system_dict;



@accepts(str, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_intermediate_model_prefix(value, system_dict):
    '''
    Set a prefix to names of intermediate models being saved

    Args:
        value (str): Appends a prefix to intermediate weights
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    system_dict["training"]["settings"]["intermediate_model_prefix"] = value;
    return system_dict;