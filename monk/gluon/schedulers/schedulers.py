from gluon.schedulers.imports import *
from system.imports import *


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def scheduler_fixed(system_dict):
    system_dict["local"]["learning_rate_scheduler"] = None;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] = "fixed";
    return system_dict;


@accepts(dict, int,  gamma=[float, int], last_epoch=int, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def scheduler_step(system_dict, step_size, gamma=0.1, last_epoch=-1):
    system_dict["local"]["learning_rate_scheduler"] = "steplr";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] = "steplr";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["step_size"] = step_size;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"] = gamma;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["last_epoch"] = last_epoch;     
    return system_dict;



@accepts(dict, [list, int], gamma=[float, int], last_epoch=int, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def scheduler_multistep(system_dict, milestones, gamma=0.1, last_epoch=-1):
    system_dict["local"]["learning_rate_scheduler"] = "multisteplr";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] = "multisteplr";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["milestones"] = milestones;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"] = gamma;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["last_epoch"] = last_epoch;
    return system_dict;


