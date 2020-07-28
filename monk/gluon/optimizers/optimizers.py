from monk.gluon.optimizers.imports import *
from monk.system.imports import *



@accepts(dict, [int, float], momentum=[int, float], momentum_dampening_rate=[int, float], weight_decay=[int, float], 
    clipnorm=[int, float], clipvalue=[int, float], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def sgd(system_dict, learning_rate, momentum=0, weight_decay=0, momentum_dampening_rate=0, clipnorm=0.0, clipvalue=0.0):
    '''
    Select stochastic gradient descent optimizer

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        learning_rate (float): Initial base learning rate
        momentum (float): Momentum value for driving the weights towards minima
        weight_decay (float): Value for regularizing weights post every update
        momentum_dampening_rate (float): Reduction rate for momentum
        clipnorm (float): Gradient clipping factor
        clipvalue (float): Value for clipping

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = "sgd";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "sgd";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"] = momentum;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum_dampening_rate"] = momentum_dampening_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"] = clipnorm;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"] = clipvalue;
    return system_dict;


@accepts(dict, [int, float], momentum=[int, float], momentum_dampening_rate=[int, float], weight_decay=[int, float], 
    clipnorm=[int, float], clipvalue=[int, float], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def nesterov_sgd(system_dict, learning_rate, momentum=0, weight_decay=0, momentum_dampening_rate=0, clipnorm=0.0, clipvalue=0.0):
    '''
    Select stochastic gradient descent optimizer with nesterov acceleration

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        learning_rate (float): Initial base learning rate
        momentum (float): Momentum value for driving the weights towards minima
        weight_decay (float): Value for regularizing weights post every update
        momentum_dampening_rate (float): Reduction rate for momentum
        clipnorm (float): Gradient clipping factor
        clipvalue (float): Value for clipping

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = "nesterov_sgd";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "nesterov_sgd";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"] = momentum;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum_dampening_rate"] = momentum_dampening_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"] = clipnorm;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"] = clipvalue;
    return system_dict;



@accepts(dict, [int, float], decay_rate=[int, float], epsilon=[int, float], weight_decay=[int, float], 
    clipnorm=[int, float], clipvalue=[int, float], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def rmsprop(system_dict, learning_rate, decay_rate=0.99, epsilon=1e-08, weight_decay=0, 
    clipnorm=0.0, clipvalue=0.0):
    '''
    Select root mean score prop optimizer

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        learning_rate (float): Initial base learning rate
        decay_rate (float): A decay factor of moving average over past squared gradient.
        epsilon (float): A value to avoid division by zero
        weight_decay (float): Value for regularizing weights post every update
        clipnorm (float): Gradient clipping factor
        clipvalue (float): Value for clipping

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = "rmsprop";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "rmsprop";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["decay_rate"] = decay_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"] = clipnorm;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"] = clipvalue;
    return system_dict;



@accepts(dict, [int, float], decay_rate=[int, float], epsilon=[int, float], weight_decay=[int, float], 
    momentum=[int, float], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def momentum_rmsprop(system_dict, learning_rate, decay_rate=0.99, epsilon=1e-08, weight_decay=0, 
    momentum=0.9):
    '''
    Select root mean score prop optimizer with momentum

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        learning_rate (float): Initial base learning rate
        decay_rate (float): A decay factor of moving average over past squared gradient.
        epsilon (float): A value to avoid division by zero
        weight_decay (float): Value for regularizing weights post every update
        momentum (float): Momentum value for driving the weights towards minima
        clipnorm (float): Gradient clipping factor
        clipvalue (float): Value for clipping

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = "rmsprop";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "rmsprop";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["decay_rate"] = decay_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"] = momentum;
    return system_dict;



@accepts(dict, [int, float], beta1=[int, float], beta2=[int, float], epsilon=[int, float], weight_decay=[int, float], amsgrad=bool, 
    clipnorm=[int, float], clipvalue=[int, float], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def adam(system_dict, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, weight_decay=0, amsgrad=False, clipnorm=0.0, clipvalue=0.0):
    '''
    Select ADAM optimizer

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        learning_rate (float): Initial base learning rate
        beta1 (float): Exponential decay rate for first momentum estimates
        beta2 (float): Exponential decay rate for first second estimates
        weight_decay (float): Value for regularizing weights post every update
        amsgrad (bool): If True, AMSGrad variant of this algorithm is used
        epsilon (float): A value to avoid division by zero
        clipnorm (float): Gradient clipping factor
        clipvalue (float): Value for clipping

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = "adam";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "adam";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"] = beta1;
    system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"] = beta2;
    system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"] = amsgrad;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"] = clipnorm;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"] = clipvalue;
    return system_dict;


@accepts(dict, [int, float], beta1=[int, float], beta2=[int, float], epsilon=[int, float], weight_decay=[int, float], 
    clipnorm=[int, float], clipvalue=[int, float], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def adamax(system_dict, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, weight_decay=0, clipnorm=0.0, clipvalue=0.0):
    '''
    Select Adamax optimizer

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        learning_rate (float): Initial base learning rate
        beta1 (float): Exponential decay rate for first momentum estimates
        beta2 (float): Exponential decay rate for first second estimates
        weight_decay (float): Value for regularizing weights post every update
        epsilon (float): A value to avoid division by zero
        clipnorm (float): Gradient clipping factor
        clipvalue (float): Value for clipping

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = "adamax";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "adamax";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"] = beta1;
    system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"] = beta2;
    system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"] = clipnorm;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"] = clipvalue;
    return system_dict;



@accepts(dict, [int, float], rho=[int, float], epsilon=[int, float], weight_decay=[int, float], 
    clipnorm=[int, float], clipvalue=[int, float], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def adadelta(system_dict, learning_rate, rho=0.9, epsilon=1e-06, weight_decay=0, clipnorm=0.0, clipvalue=0.0):
    '''
    Select Adadelta optimizer

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        learning_rate (float): Initial base learning rate
        rho (float): Exponential decay rate for momentum estimates
        weight_decay (float): Value for regularizing weights post every update
        epsilon (float): A value to avoid division by zero
        clipnorm (float): Gradient clipping factor
        clipvalue (float): Value for clipping

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = "adadelta";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "adadelta";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["rho"] = rho;
    system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"] = clipnorm;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"] = clipvalue;
    return system_dict;



@accepts(dict, [int, float], learning_rate_decay=[int, float], weight_decay=[int, float], epsilon=[int, float], 
    clipnorm=[int, float], clipvalue=[int, float], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def adagrad(system_dict, learning_rate, learning_rate_decay=0, weight_decay=0, epsilon=0, clipnorm=0.0, clipvalue=0.0):
    '''
    Select Adagrad optimizer

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        learning_rate (float): Initial base learning rate
        learning_rate_decay (float): Learning rate decay factor
        weight_decay (float): Value for regularizing weights post every update
        epsilon (float): A value to avoid division by zero
        clipnorm (float): Gradient clipping factor
        clipvalue (float): Value for clipping

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = "adagrad";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "adagrad";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"] = learning_rate_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"] = clipnorm;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"] = clipvalue;
    return system_dict;



@accepts(dict, [int, float], beta1=[int, float], beta2=[int, float], epsilon=[int, float], weight_decay=[int, float], amsgrad=bool, 
    momentum_decay=[int, float], clipnorm=[int, float], clipvalue=[int, float], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def nesterov_adam(system_dict, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, weight_decay=0, amsgrad=False,
    momentum_decay=0.004, clipnorm=0.0, clipvalue=0.0):
    '''
    Select ADAM optimizer with nesterov momentum acceleration

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        learning_rate (float): Initial base learning rate
        beta1 (float): Exponential decay rate for first momentum estimates
        beta2 (float): Exponential decay rate for first second estimates
        weight_decay (float): Value for regularizing weights post every update
        amsgrad (bool): If True, AMSGrad variant of this algorithm is used
        epsilon (float): A value to avoid division by zero
        clipnorm (float): Gradient clipping factor
        clipvalue (float): Value for clipping

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = "nadam";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "nadam";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"] = beta1;
    system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"] = beta2;
    system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"] = amsgrad;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum_decay"] = momentum_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"] = clipnorm;
    system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"] = clipvalue;
    return system_dict;



@accepts(dict, [int, float], momentum=[int, float], weight_decay=[int, float], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def signum(system_dict, learning_rate, momentum=0, weight_decay=0, nesterov=False):
    '''
    Select SIGNUM optimizer

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        learning_rate (float): Initial base learning rate
        momentum (float): Momentum value for driving the weights towards minima
        weight_decay (float): Value for regularizing weights post every update

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["optimizer"] = "signum";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "signum";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"] = momentum;
    return system_dict;


@accepts(dict, [int, float], betas=tuple, epsilon=[int, float], weight_decay=[int, float], amsgrad=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def ftml(system_dict, learning_rate, betas=(0.9, 0.999), epsilon=1e-08, weight_decay=0, amsgrad=False):
    '''
    Inactive function
    '''
    system_dict["local"]["optimizer"] = "ftml";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "ftml";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["betas"] = betas;
    system_dict["hyper-parameters"]["optimizer"]["params"]["eps"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"] = amsgrad;
    return system_dict;