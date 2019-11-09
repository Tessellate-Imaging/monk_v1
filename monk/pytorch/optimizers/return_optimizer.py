from pytorch.optimizers.imports import *
from system.imports import *

@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def load_optimizer(system_dict):
    optimizer = system_dict["local"]["optimizer"];
    learning_rate = system_dict["hyper-parameters"]["learning_rate"];

    if(optimizer == "adadelta"):
        system_dict["local"]["optimizer"] = torch.optim.Adadelta(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            rho=system_dict["hyper-parameters"]["optimizer"]["params"]["rho"], 
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"]);

    elif(optimizer == "adagrad"):
        system_dict["local"]["optimizer"] = torch.optim.Adagrad(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            lr_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            initial_accumulator_value=system_dict["hyper-parameters"]["optimizer"]["params"]["initial_accumulator_value"]);

    elif(optimizer == "adam"):
        system_dict["local"]["optimizer"] = torch.optim.Adam(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            betas=system_dict["hyper-parameters"]["optimizer"]["params"]["betas"], 
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            amsgrad=system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"]);

    elif(optimizer == "adamw"):
        system_dict["local"]["optimizer"] = torch.optim.AdamW(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            betas=system_dict["hyper-parameters"]["optimizer"]["params"]["betas"], 
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            amsgrad=system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"]);

    elif(optimizer == "sparseadam"):
        system_dict["local"]["optimizer"] = torch.optim.SparseAdam(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            betas=system_dict["hyper-parameters"]["optimizer"]["params"]["betas"], 
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"]);

    elif(optimizer == "adamax"):
        system_dict["local"]["optimizer"] = torch.optim.Adamax(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            betas=system_dict["hyper-parameters"]["optimizer"]["params"]["betas"], 
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"]);

    elif(optimizer == "asgd"):
        system_dict["local"]["optimizer"] = torch.optim.ASGD(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            lambd=system_dict["hyper-parameters"]["optimizer"]["params"]["lambd"], 
            alpha=system_dict["hyper-parameters"]["optimizer"]["params"]["alpha"], 
            t0=system_dict["hyper-parameters"]["optimizer"]["params"]["t0"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"]);

    elif(optimizer == "rmsprop"):
        system_dict["local"]["optimizer"] = torch.optim.RMSprop(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            alpha=system_dict["hyper-parameters"]["optimizer"]["params"]["alpha"], 
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["eps"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"], 
            centered=system_dict["hyper-parameters"]["optimizer"]["params"]["centered"]);

    elif(optimizer == "rprop"):
        system_dict["local"]["optimizer"] = torch.optim.Rprop(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            etas=system_dict["hyper-parameters"]["optimizer"]["params"]["etas"], 
            step_sizes=system_dict["hyper-parameters"]["optimizer"]["params"]["step_sizes"]);
        
    elif(optimizer == "sgd"):
        system_dict["local"]["optimizer"] = torch.optim.SGD(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"], 
            dampening=system_dict["hyper-parameters"]["optimizer"]["params"]["dampening"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            nesterov=system_dict["hyper-parameters"]["optimizer"]["params"]["nesterov"]);


    return system_dict;