from pytorch.models.imports import *
from system.imports import *
from pytorch.models.models import combined_list_lower

@accepts(str, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_model_name(name, system_dict):
    if(name not in combined_list_lower):
        msg = "Model name {} not in {}".format(name, combined_list_lower);
        raise ConstraintError(msg);
    system_dict["model"]["params"]["model_name"] = name;
    return system_dict;



@accepts(bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_device(value, system_dict):
    GPUs = GPUtil.getGPUs()
    if(value and len(GPUs)==0):
        msg = "GPU not accessible yet requested."
        ConstraintWarning(msg)
        system_dict["model"]["params"]["use_gpu"] = False;
    else:
        system_dict["model"]["params"]["use_gpu"] = value;
    return system_dict;


@accepts(bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_pretrained(value, system_dict):
    system_dict["model"]["params"]["use_pretrained"] = value;
    return system_dict;


@accepts(bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_freeze_base_network(value, system_dict):
    system_dict["model"]["params"]["freeze_base_network"] = value;
    return system_dict;


@accepts([str, list], dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_model_path(path, system_dict):
    system_dict["model"]["params"]["model_path"] = path;
    return system_dict;