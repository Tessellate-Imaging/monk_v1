from tf_keras.schedulers.imports import *
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


@accepts(dict, [float, int], last_epoch=int, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def scheduler_exponential(system_dict, gamma, last_epoch=-1):
    system_dict["local"]["learning_rate_scheduler"] = "exponential";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] = "exponential";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"] = gamma;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["last_epoch"] = last_epoch;
    return system_dict;


@accepts(dict, mode=str, factor=[float, int], patience=int, verbose=bool, threshold=[float, int],
    threshold_mode=str, cooldown=int, min_lr=[float, list, int], epsilon=float, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def scheduler_plateau(system_dict, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, 
    threshold_mode='rel', cooldown=0, min_lr=0, epsilon=1e-08):
    system_dict["local"]["learning_rate_scheduler"] = "reduceonplateaulr";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] = "reduceonplateaulr";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["mode"] = mode;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["factor"] = factor;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["patience"] = patience;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["verbose"] = verbose;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["threshold"] = threshold;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["threshold_mode"] = threshold_mode;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["cooldown"] = cooldown;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["min_lr"] = min_lr;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["epsilon"] = epsilon;
    return system_dict;