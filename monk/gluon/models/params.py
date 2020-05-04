from gluon.models.imports import *
from system.imports import *
from gluon.models.models import combined_list_lower

@accepts(str, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_model_name(name, system_dict):
    '''
    Set base model name for transfer learning.

    Args:
        name (str): Select from available models. Check via List_Models() function
        system_dict (dict): System Dictionary

    Returns:
        dict: Updated system dictionary.
    '''
    if(name not in combined_list_lower):
        msg = "Model name {} not in {}".format(name, combined_list_lower);
        raise ConstraintError(msg);
    system_dict["model"]["params"]["model_name"] = name;
    return system_dict;



@accepts(bool, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_device(value, system_dict):
    '''
    Set whether to use gpu or not

    Args:
        value (bool): If set as True, uses GPU
        system_dict (dict): System Dictionary

    Returns:
        dict: Updated system dictionary.
    '''
    if(value and mx.context.num_gpus()==0):
        msg = "GPU not accessible yet requested."
        ConstraintWarning(msg)
        system_dict["model"]["params"]["use_gpu"] = False;
    else:
        system_dict["model"]["params"]["use_gpu"] = value;
    return system_dict;


@accepts(bool, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_pretrained(value, system_dict):
    '''
    Set whether to use pretrained models or randomly initialized weights

    Args:
        value (bool): If set as True, use weights trained on imagenet and coco like dataset
                                    Else, use randomly initialized weights
        system_dict (dict): System Dictionary

    Returns:
        dict: Updated system dictionary.
    '''
    system_dict["model"]["params"]["use_pretrained"] = value;
    return system_dict;


@accepts(bool, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_freeze_base_network(value, system_dict):
    '''
    Set whether to freeze base network or not

    Args:
        value (bool): If set as True, then base network's weights are freezed (cannot be trained)
        system_dict (dict): System Dictionary

    Returns:
        dict: Updated system dictionary.
    '''
    system_dict["model"]["params"]["freeze_base_network"] = value;
    return system_dict;


@accepts([str, list], dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_model_path(path, system_dict):
    '''
    Set path to custom weights for model

    Args:
        path (str): Path to custom model weights for initialization.
        system_dict (dict): System Dictionary
        
    Returns:
        dict: Updated system dictionary.
    '''
    system_dict["model"]["params"]["model_path"] = path;
    return system_dict;