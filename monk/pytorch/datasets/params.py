from pytorch.datasets.imports import *
from system.imports import *


@accepts(int, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_input_size(input_size, system_dict):
    system_dict["dataset"]["params"]["input_size"] = input_size;
    return system_dict;

@accepts(int, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_batch_size(batch_size, system_dict):
    system_dict["dataset"]["params"]["batch_size"] = batch_size;
    return system_dict;
    
@accepts(bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_data_shuffle(value, system_dict):
    train_shuffle = value;
    val_shuffle = value;
    system_dict["dataset"]["params"]["train_shuffle"] = train_shuffle;
    system_dict["dataset"]["params"]["val_shuffle"] = val_shuffle;
    return system_dict;
    
@accepts(int, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_num_processors(num_workers, system_dict):
    system_dict["dataset"]["params"]["num_workers"] = num_workers;
    return system_dict;

@accepts(bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_weighted_sampling(sample, system_dict):
    system_dict["dataset"]["params"]["weighted_sample"] = sample;
    return system_dict;