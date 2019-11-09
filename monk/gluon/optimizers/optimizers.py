from gluon.optimizers.imports import *
from system.imports import *



@accepts(dict, [int, float], momentum=[int, float], dampening=[int, float], weight_decay=[int, float], nesterov=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def sgd(system_dict, learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    system_dict["local"]["optimizer"] = "sgd";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "sgd";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["dampening"] = dampening;
    system_dict["hyper-parameters"]["optimizer"]["params"]["nesterov"] = nesterov;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"] = momentum;
    return system_dict;


@accepts(dict, [int, float], momentum=[int, float], dampening=[int, float], weight_decay=[int, float], nesterov=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def nag(system_dict, learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    system_dict["local"]["optimizer"] = "nag";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "nag";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["dampening"] = dampening;
    system_dict["hyper-parameters"]["optimizer"]["params"]["nesterov"] = nesterov;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"] = momentum;
    return system_dict;



@accepts(dict, [int, float], alpha=[int, float], epsilon=[int, float], weight_decay=[int, float], momentum=[int, float], centered=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def rmsprop(system_dict, learning_rate, alpha=0.99, epsilon=1e-08, weight_decay=0, momentum=0, centered=False):
    system_dict["local"]["optimizer"] = "rmsprop";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "rmsprop";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["eps"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["alpha"] = alpha;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"] = momentum;
    system_dict["hyper-parameters"]["optimizer"]["params"]["centered"] = centered;
    return system_dict;



@accepts(dict, [int, float], betas=tuple, epsilon=[int, float], weight_decay=[int, float], amsgrad=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def adam(system_dict, learning_rate, betas=(0.9, 0.999), epsilon=1e-08, weight_decay=0, amsgrad=False):
    system_dict["local"]["optimizer"] = "adam";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "adam";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["betas"] = betas;
    system_dict["hyper-parameters"]["optimizer"]["params"]["eps"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"] = amsgrad;
    return system_dict;



@accepts(dict, [int, float], learning_rate_decay=[int, float], weight_decay=[int, float], initial_accumulator_value=[int, float], post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def adagrad(system_dict, learning_rate, learning_rate_decay=0, weight_decay=0, initial_accumulator_value=0):
    system_dict["local"]["optimizer"] = "adagrad";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "adagrad";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"] = learning_rate_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["initial_accumulator_value"] = initial_accumulator_value;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    return system_dict;



@accepts(dict, [int, float], rho=[int, float], epsilon=[int, float], weight_decay=[int, float], post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def adadelta(system_dict, learning_rate, rho=0.9, epsilon=1e-06, weight_decay=0):
    system_dict["local"]["optimizer"] = "adadelta";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "adadelta";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["rho"] = rho;
    system_dict["hyper-parameters"]["optimizer"]["params"]["eps"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    return system_dict;



@accepts(dict, [int, float], betas=tuple, epsilon=[int, float], weight_decay=[int, float], post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def adamax(system_dict, learning_rate, betas=(0.9, 0.999), epsilon=1e-08, weight_decay=0.01):
    system_dict["local"]["optimizer"] = "adamax";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "adamax";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["betas"] = betas;
    system_dict["hyper-parameters"]["optimizer"]["params"]["eps"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    return system_dict;
    


@accepts(dict, [int, float], betas=tuple, epsilon=[int, float], weight_decay=[int, float], amsgrad=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def nadam(system_dict, learning_rate, betas=(0.9, 0.999), epsilon=1e-08, weight_decay=0, amsgrad=False):
    system_dict["local"]["optimizer"] = "nadam";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "nadam";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["betas"] = betas;
    system_dict["hyper-parameters"]["optimizer"]["params"]["eps"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"] = amsgrad;
    return system_dict;



@accepts(dict, [int, float], momentum=[int, float], dampening=[int, float], weight_decay=[int, float], nesterov=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def signum(system_dict, learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    system_dict["local"]["optimizer"] = "signum";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "signum";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["dampening"] = dampening;
    system_dict["hyper-parameters"]["optimizer"]["params"]["nesterov"] = nesterov;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"] = momentum;
    return system_dict;




@accepts(dict, [int, float], betas=tuple, epsilon=[int, float], weight_decay=[int, float], amsgrad=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def ftml(system_dict, learning_rate, betas=(0.9, 0.999), epsilon=1e-08, weight_decay=0, amsgrad=False):
    system_dict["local"]["optimizer"] = "ftml";
    system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["name"] = "ftml";
    system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
    system_dict["hyper-parameters"]["optimizer"]["params"]["betas"] = betas;
    system_dict["hyper-parameters"]["optimizer"]["params"]["eps"] = epsilon;
    system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"] = weight_decay;
    system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"] = amsgrad;
    return system_dict;