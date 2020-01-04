from gluon.optimizers.imports import *
from system.imports import *

@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def load_optimizer(system_dict):
    optimizer = system_dict["local"]["optimizer"];
    learning_rate_scheduler = system_dict["local"]["learning_rate_scheduler"];
    learning_rate = system_dict["hyper-parameters"]["learning_rate"];

    if(optimizer == "sgd"):
        system_dict["local"]["optimizer"] = mx.optimizer.SGD(
            learning_rate=learning_rate,
            wd=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"],
            lr_scheduler=learning_rate_scheduler, 
            momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"]);
    
    elif(optimizer == "nesterov_sgd"):
        system_dict["local"]["optimizer"] = mx.optimizer.NAG(
            learning_rate=learning_rate, 
            wd=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            lr_scheduler=learning_rate_scheduler, 
            momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"]);

    elif(optimizer == "rmsprop"):
        system_dict["local"]["optimizer"] = mx.optimizer.RMSProp(
            learning_rate=learning_rate, 
            wd=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            lr_scheduler=learning_rate_scheduler,
            gamma1=system_dict["hyper-parameters"]["optimizer"]["params"]["decay_rate"], 
            gamma2=0.0, 
            epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
            centered=False);

    elif(optimizer == "momentum_rmsprop"):
        system_dict["local"]["optimizer"] = mx.optimizer.RMSProp(
            learning_rate=learning_rate, 
            wd=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            lr_scheduler=learning_rate_scheduler,
            gamma1=system_dict["hyper-parameters"]["optimizer"]["params"]["decay_rate"], 
            gamma2=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"], 
            epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
            centered=True);


    elif(optimizer == "adam"):
        system_dict["local"]["optimizer"] = mx.optimizer.Adam(
            learning_rate=learning_rate, 
            wd=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            lr_scheduler=learning_rate_scheduler,
            beta1=system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"], 
            beta2=system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"], 
            epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"]);


    elif(optimizer == "adamax"):
        system_dict["local"]["optimizer"] = mx.optimizer.Adamax(
            learning_rate=learning_rate, 
            wd=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            lr_scheduler=learning_rate_scheduler,
            beta1=system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"], 
            beta2=system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"]);


    elif(optimizer == "adadelta"):
        system_dict["local"]["optimizer"] = mx.optimizer.AdaDelta(
            learning_rate=learning_rate, 
            wd=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            lr_scheduler=learning_rate_scheduler,
            rho=system_dict["hyper-parameters"]["optimizer"]["params"]["rho"], 
            epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"]);


    elif(optimizer == "adagrad"):
        system_dict["local"]["optimizer"] = mx.optimizer.AdaGrad(
            learning_rate=learning_rate, 
            wd=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            lr_scheduler=learning_rate_scheduler,
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"]);

    elif(optimizer == "nadam"):
        system_dict["local"]["optimizer"] = mx.optimizer.Nadam(
            learning_rate=learning_rate, 
            wd=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            lr_scheduler=learning_rate_scheduler,
            beta1=system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"], 
            beta2=system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"], 
            epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"],
            schedule_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum_decay"]
            );

    elif(optimizer == "signum"):
        system_dict["local"]["optimizer"] = mx.optimizer.Signum(
            learning_rate=learning_rate, 
            wd_lh=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            lr_scheduler=learning_rate_scheduler, 
            momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"]);

    return system_dict;