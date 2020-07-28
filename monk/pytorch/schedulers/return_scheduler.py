from monk.pytorch.schedulers.imports import *
from monk.system.imports import *


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def load_scheduler(system_dict):
    '''
    Load schedulers for training state

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    learning_rate_scheduler = system_dict["local"]["learning_rate_scheduler"];
    optimizer = system_dict["local"]["optimizer"];
    
    if(learning_rate_scheduler == "steplr"):
        system_dict["local"]["learning_rate_scheduler"] = torch.optim.lr_scheduler.StepLR(
                                                            optimizer,
                                                            system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["step_size"], 
                                                            gamma=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"], 
                                                            last_epoch=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["last_epoch"]);

    elif(learning_rate_scheduler == "multisteplr"):
        system_dict["local"]["learning_rate_scheduler"] = torch.optim.lr_scheduler.MultiStepLR(
                                                            optimizer, 
                                                            system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["milestones"], 
                                                            gamma=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"], 
                                                            last_epoch=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["last_epoch"]);

    elif(learning_rate_scheduler == "exponentiallr"):
        system_dict["local"]["learning_rate_scheduler"] = torch.optim.lr_scheduler.ExponentialLR(
                                                            optimizer, 
                                                            system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"], 
                                                            last_epoch=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["last_epoch"]);

    elif(learning_rate_scheduler == "reduceonplateaulr"):
        system_dict["local"]["learning_rate_scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            optimizer, 
                                                            mode=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["mode"], 
                                                            factor=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["factor"], 
                                                            patience=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["patience"],
                                                            verbose=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["verbose"], 
                                                            threshold=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["threshold"], 
                                                            threshold_mode=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["threshold_mode"], 
                                                            cooldown=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["cooldown"], 
                                                            min_lr=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["min_lr"], 
                                                            eps=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["epsilon"]);

    return system_dict;


