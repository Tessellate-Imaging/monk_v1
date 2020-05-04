from tf_keras_1.schedulers.imports import *
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


@accepts(dict, [float, int], last_epoch=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def scheduler_exponential(system_dict, gamma, last_epoch=-1):
    '''
    Set learning rate to decrease exponentially every step

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        gamma (str): Reduction multiplier for reducing learning rate post every step
        last_epoch (int): Set this epoch to a level post which learning rate will not be decreased

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["learning_rate_scheduler"] = "exponential";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] = "exponential";
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"] = gamma;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["last_epoch"] = last_epoch;
    return system_dict;


@accepts(dict, mode=str, factor=[float, int], patience=int, verbose=bool, threshold=[float, int],
    threshold_mode=str, cooldown=int, min_lr=[float, list, int], epsilon=float, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def scheduler_plateau(system_dict, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, 
    threshold_mode='rel', cooldown=0, min_lr=0, epsilon=1e-08):
    '''
    Set learning rate to decrease if a metric (loss) stagnates in a plateau

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        mode (str): Either of 
                    - 'min' : lr will be reduced when the quantity monitored (loss) has stopped decreasing; 
                    - 'max' : lr reduced when the quantity monitored (accuracy) has stopped increasing. 
        factor (float): Reduction multiplier for reducing learning rate post every step
        patience (int): Number of epochs to wait before reducing learning rate
        verbose (bool): If True, all computations and wait times are printed
        threshold (float): Preset fixed to 0.0001
        threshold_mode (str): Preset fixed to 'rel' mode
        cooldown (int): Number of epochs to wait before actually applying the scheduler post the actual designated step
        min_lr (float): Set minimum learning rate, post which it will not be decreased
        epsilon (float): A small value to avoid divison by zero.
        last_epoch (int): Set this epoch to a level post which learning rate will not be decreased

    Returns:
        dict: updated system dict
    '''
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