from pytorch.training.imports import *
from system.imports import *


@accepts(int, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_num_epochs(num_epochs, system_dict):
    system_dict["hyper-parameters"]["num_epochs"] = num_epochs;
    return system_dict;



@accepts(bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_display_progress_realtime(value, system_dict):
    system_dict["training"]["settings"]["display_progress_realtime"] = value;
    return system_dict;


@accepts(bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_display_progress(value, system_dict):
    system_dict["training"]["settings"]["display_progress"] = value;
    return system_dict;


@accepts(bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_save_intermediate_models(value, system_dict):
    system_dict["training"]["settings"]["save_intermediate_models"] = value;
    return system_dict;


@accepts(bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_save_training_logs(value, system_dict):
    system_dict["training"]["settings"]["save_training_logs"] = value;
    return system_dict;


@accepts(str, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_intermediate_model_prefix(value, system_dict):
    system_dict["training"]["settings"]["intermediate_model_prefix"] = value;
    return system_dict;