from monk.gluon.datasets.imports import *
from monk.system.imports import *


@accepts([int, tuple], dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_input_size(input_size, system_dict):
    '''
    Set Input data size

    Args:
        input_size (int, tuple): Single integer in case of square shaped image
                                  Tuple representing (width, height)
        system_dict (dict): System Dictionary
    Returns:
        dict: Updated system dictionary.
    '''
    system_dict["dataset"]["params"]["input_size"] = input_size;
    return system_dict;


@accepts(int, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_batch_size(batch_size, system_dict):
    '''
    Set Input batch size

    Args:
        batch_size (int): Batch sizes for sampling input data during training
        system_dict (dict): System Dictionary
    Returns:
        dict: Updated system dictionary.
    '''
    system_dict["dataset"]["params"]["batch_size"] = batch_size;
    return system_dict;
    

@accepts(bool, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_data_shuffle(value, system_dict):
    '''
    Set weather to shuffle data

    Args:
        value (bool): If True data is shuffled before sampling into batches
        system_dict (dict): System Dictionary
    Returns:
        dict: Updated system dictionary.
    '''
    train_shuffle = value;
    val_shuffle = value;
    system_dict["dataset"]["params"]["train_shuffle"] = train_shuffle;
    system_dict["dataset"]["params"]["val_shuffle"] = val_shuffle;
    return system_dict;
    

@accepts(int, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_num_processors(num_workers, system_dict):
    '''
    Set Number of processors for data sampling

    Args:
        num_workers (int): Max number of CPUs to be used
        system_dict (dict): System Dictionary
    Returns:
        dict: Updated system dictionary.
    '''
    system_dict["dataset"]["params"]["num_workers"] = num_workers;
    return system_dict;


@accepts(bool, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_weighted_sampling(sample, system_dict):
    '''
            
    '''
    system_dict["dataset"]["params"]["weighted_sample"] = sample;
    return system_dict;