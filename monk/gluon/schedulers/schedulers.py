from gluon.schedulers.imports import *
from system.imports import *


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def scheduler_fixed(system_dict):
    '''
    Set learning rate fixed

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["learning_rate_scheduler"] = None;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] = "fixed";
    return system_dict;


@accepts(dict, int,  gamma=[float, int], last_epoch=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def scheduler_step(system_dict, step_size, gamma=0.1, last_epoch=-1):
    '''
    Set learning rate to decrease in regular steps

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        step_size (int): Step interval for decreasing learning rate
        gamma (str): Reduction multiplier for reducing learning rate post every step
        last_epoch (int): Set this epoch to a level post which learning rate will not be decreased

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["learning_rate_scheduler"] = "steplr";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] = "steplr";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["step_size"] = step_size;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"] = gamma;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["last_epoch"] = last_epoch;     
    return system_dict;



@accepts(dict, [list, int], gamma=[float, int], last_epoch=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def scheduler_multistep(system_dict, milestones, gamma=0.1, last_epoch=-1):
    '''
    Set learning rate to decrease in irregular steps

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        milestones (list): List of epochs at which learning rate is to be decreased
        gamma (str): Reduction multiplier for reducing learning rate post every step
        last_epoch (int): Dummy variable

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["learning_rate_scheduler"] = "multisteplr";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] = "multisteplr";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["milestones"] = milestones;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"] = gamma;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["last_epoch"] = last_epoch;
    return system_dict;


