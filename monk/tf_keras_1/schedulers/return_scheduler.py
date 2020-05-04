from tf_keras_1.schedulers.imports import *
from system.imports import *
from tf_keras_1.schedulers.common import StepDecay
from tf_keras_1.schedulers.common import PolynomialDecay


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
    learning_rate = system_dict["hyper-parameters"]["learning_rate"];

    if(learning_rate_scheduler == "steplr"):
        system_dict["local"]["learning_rate_scheduler"] = StepDecay(
                initAlpha=learning_rate, 
                factor=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"], 
                dropEvery=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["step_size"]);

    elif(learning_rate_scheduler == "exponentiallr"):
        system_dict["local"]["learning_rate_scheduler"] = PolynomialDecay(
                maxEpochs=100, 
                initAlpha=learning_rate, 
                power=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["gamma"]);

    elif(learning_rate_scheduler == "reduceonplateaulr"):
        system_dict["local"]["learning_rate_scheduler"] = krc.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["factor"], 
                patience=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["patience"], 
                verbose=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["verbose"], 
                mode=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["mode"], 
                min_delta=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["threshold"], 
                cooldown=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["cooldown"], 
                min_lr=system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]["min_lr"]);


    return system_dict;


