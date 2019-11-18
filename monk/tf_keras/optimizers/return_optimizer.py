from tf_keras.optimizers.imports import *
from system.imports import *

@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def load_optimizer(system_dict):
    optimizer = system_dict["local"]["optimizer"];
    learning_rate = system_dict["hyper-parameters"]["learning_rate"];

    if(optimizer == "adadelta"):
        system_dict["local"]["optimizer"] = kro.Adadelta(
                lr=learning_rate, 
                rho=system_dict["hyper-parameters"]["optimizer"]["params"]["rho"], 
                epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"]);

    elif(optimizer == "adagrad"):
        system_dict["local"]["optimizer"] = kro.Adagrad(
                lr=learning_rate, 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"]);

    elif(optimizer == "adam"):
        system_dict["local"]["optimizer"] = kro.Adam(
                lr=learning_rate, 
                beta_1=system_dict["hyper-parameters"]["optimizer"]["params"]["betas"][0], 
                beta_2=system_dict["hyper-parameters"]["optimizer"]["params"]["betas"][1], 
                epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"], 
                amsgrad=system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"]);

    elif(optimizer == "adamax"):
        system_dict["local"]["optimizer"] = kro.Adamax(
                lr=learning_rate, 
                beta_1=system_dict["hyper-parameters"]["optimizer"]["params"]["betas"][0], 
                beta_2=system_dict["hyper-parameters"]["optimizer"]["params"]["betas"][1], 
                epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"]);

    elif(optimizer == "rmsprop"):
        system_dict["local"]["optimizer"] = kro.RMSprop(
                lr=learning_rate, 
                rho=system_dict["hyper-parameters"]["optimizer"]["params"]["alpha"], 
                epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"]);

    elif(optimizer == "sgd"):
        system_dict["local"]["optimizer"] = kro.SGD(
                lr=learning_rate, 
                momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"], 
                nesterov=system_dict["hyper-parameters"]["optimizer"]["params"]["nesterov"]);

    elif(optimizer == "nadam"):
        system_dict["local"]["optimizer"] = kro.Nadam(
                lr=learning_rate, 
                beta_1=system_dict["hyper-parameters"]["optimizer"]["params"]["betas"][0], 
                beta_2=system_dict["hyper-parameters"]["optimizer"]["params"]["betas"][1], 
                epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"], 
                schedule_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"]);

    return system_dict;