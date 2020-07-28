from monk.pytorch.optimizers.imports import *
from monk.system.imports import *

@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def load_optimizer(system_dict):
    '''
    Load Optimizers in training states

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    optimizer = system_dict["local"]["optimizer"];
    learning_rate = system_dict["hyper-parameters"]["learning_rate"];

    
    if(optimizer == "sgd"):
        system_dict["local"]["optimizer"] = torch.optim.SGD(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"], 
            dampening=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum_dampening_rate"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            nesterov=False);

    elif(optimizer == "nesterov_sgd"):
        system_dict["local"]["optimizer"] = torch.optim.SGD(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"], 
            dampening=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum_dampening_rate"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            nesterov=False);

    elif(optimizer == "rmsprop"):
        system_dict["local"]["optimizer"] = torch.optim.RMSprop(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            alpha=system_dict["hyper-parameters"]["optimizer"]["params"]["decay_rate"], 
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            momentum=0.0, 
            centered=False);

    elif(optimizer == "momentum_rmsprop"):
        system_dict["local"]["optimizer"] = torch.optim.RMSprop(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            alpha=system_dict["hyper-parameters"]["optimizer"]["params"]["decay_rate"], 
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"], 
            centered=True);


    elif(optimizer == "adam"):
        system_dict["local"]["optimizer"] = torch.optim.Adam(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            betas=(system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"], system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"]),  
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            amsgrad=system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"]);


    elif(optimizer == "adamax"):
        system_dict["local"]["optimizer"] = torch.optim.Adamax(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            betas=(system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"], system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"]), 
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"]);


    elif(optimizer == "adamw"):
        system_dict["local"]["optimizer"] = torch.optim.AdamW(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            betas=(system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"], system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"]),  
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            amsgrad=system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"]);


    elif(optimizer == "adadelta"):
        system_dict["local"]["optimizer"] = torch.optim.Adadelta(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            rho=system_dict["hyper-parameters"]["optimizer"]["params"]["rho"], 
            eps=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"]);

    elif(optimizer == "adagrad"):
        system_dict["local"]["optimizer"] = torch.optim.Adagrad(
            system_dict["local"]["params_to_update"], 
            lr=learning_rate, 
            lr_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["lr_decay"], 
            weight_decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
            initial_accumulator_value=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"]);


    return system_dict;