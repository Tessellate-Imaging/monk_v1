from monk.tf_keras_1.optimizers.imports import *
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
        system_dict["local"]["optimizer"] = kro.SGD(
                lr=learning_rate, 
                momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
                nesterov=False,
                clipnorm=system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"],
                clipvalue=system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"]);

    elif(optimizer == "nesterov_sgd"):
        system_dict["local"]["optimizer"] = kro.SGD(
                lr=learning_rate, 
                momentum=system_dict["hyper-parameters"]["optimizer"]["params"]["momentum"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
                nesterov=True,
                clipnorm=system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"],
                clipvalue=system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"]);


    elif(optimizer == "rmsprop"):
        system_dict["local"]["optimizer"] = kro.RMSprop(
                lr=learning_rate, 
                rho=system_dict["hyper-parameters"]["optimizer"]["params"]["decay_rate"], 
                epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"],
                clipnorm=system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"],
                clipvalue=system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"]);


    elif(optimizer == "adam"):
        system_dict["local"]["optimizer"] = kro.Adam(
                lr=learning_rate, 
                beta_1=system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"], 
                beta_2=system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"], 
                epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"], 
                amsgrad=system_dict["hyper-parameters"]["optimizer"]["params"]["amsgrad"],
                clipnorm=system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"],
                clipvalue=system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"]);


    elif(optimizer == "nadam"):
        system_dict["local"]["optimizer"] = kro.Nadam(
            lr=learning_rate,
            beta_1=system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"], 
            beta_2=system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"], 
            epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"],
            clipnorm=system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"],
            clipvalue=system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"]
            );

    elif(optimizer == "adamax"):
        system_dict["local"]["optimizer"] = kro.Adamax(
                lr=learning_rate, 
                beta_1=system_dict["hyper-parameters"]["optimizer"]["params"]["beta1"], 
                beta_2=system_dict["hyper-parameters"]["optimizer"]["params"]["beta2"], 
                epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"],
                clipnorm=system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"],
                clipvalue=system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"]);

    elif(optimizer == "adadelta"):
        system_dict["local"]["optimizer"] = kro.Adadelta(
                lr=learning_rate, 
                rho=system_dict["hyper-parameters"]["optimizer"]["params"]["rho"], 
                epsilon=system_dict["hyper-parameters"]["optimizer"]["params"]["epsilon"], 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"],
                clipnorm=system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"],
                clipvalue=system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"]);


    elif(optimizer == "adagrad"):
        system_dict["local"]["optimizer"] = kro.Adagrad(
                lr=learning_rate, 
                decay=system_dict["hyper-parameters"]["optimizer"]["params"]["weight_decay"],
                clipnorm=system_dict["hyper-parameters"]["optimizer"]["params"]["clipnorm"],
                clipvalue=system_dict["hyper-parameters"]["optimizer"]["params"]["clipvalue"]);




    


    return system_dict;