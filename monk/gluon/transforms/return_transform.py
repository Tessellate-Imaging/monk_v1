from monk.gluon.transforms.imports import *
from monk.system.imports import *



@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_transform_trainval(system_dict):
    '''
    Load training and validation transforms in main state

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    if(not system_dict["local"]["normalize"]):
        if(not system_dict["local"]["applied_train_tensor"]):
            if(type(system_dict["dataset"]["params"]["input_size"]) == tuple or type(system_dict["dataset"]["params"]["input_size"]) == list):
                h = system_dict["dataset"]["params"]["input_size"][0];
                w = system_dict["dataset"]["params"]["input_size"][1];
            else:
                h = system_dict["dataset"]["params"]["input_size"];
                w = system_dict["dataset"]["params"]["input_size"];

            system_dict["local"]["transforms_train"].append(transforms.Resize(size=(w, h)));
            system_dict["local"]["transforms_train"].append(transforms.ToTensor());
            system_dict["local"]["transforms_val"].append(transforms.Resize(size=(w, h)));
            system_dict["local"]["transforms_val"].append(transforms.ToTensor());
            system_dict["local"]["applied_train_tensor"] = True;
    system_dict["local"]["data_transforms"]["train"] = transforms.Compose(system_dict["local"]["transforms_train"]);
    system_dict["local"]["data_transforms"]["val"] = transforms.Compose(system_dict["local"]["transforms_val"]);

    return system_dict;



@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_transform_test(system_dict):
    '''
    Load testing transforms in main state

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        dict: updated system dict
    '''
    if(not system_dict["local"]["normalize"]):
        if(not system_dict["local"]["applied_test_tensor"]):

            if(type(system_dict["dataset"]["params"]["input_size"]) == tuple or type(system_dict["dataset"]["params"]["input_size"]) == list):
                h = system_dict["dataset"]["params"]["input_size"][0];
                w = system_dict["dataset"]["params"]["input_size"][1];
            else:
                h = system_dict["dataset"]["params"]["input_size"];
                w = system_dict["dataset"]["params"]["input_size"];
            
            system_dict["local"]["transforms_test"].append(transforms.Resize(size=(w, h)));
            system_dict["local"]["transforms_test"].append(transforms.ToTensor());
            system_dict["local"]["applied_test_tensor"] = True;
    system_dict["local"]["data_transforms"]["test"] = transforms.Compose(system_dict["local"]["transforms_test"]);

    return system_dict;