from pytorch.transforms.imports import *
from system.imports import *


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_transform_trainval(system_dict):
    if(not system_dict["local"]["normalize"]):
        if(not system_dict["local"]["applied_train_tensor"]):
            system_dict["local"]["transforms_train"].append(transforms.Resize(size=(system_dict["dataset"]["params"]["input_size"], system_dict["dataset"]["params"]["input_size"])));
            system_dict["local"]["transforms_train"].append(transforms.ToTensor());
            system_dict["local"]["transforms_val"].append(transforms.Resize(size=(system_dict["dataset"]["params"]["input_size"], system_dict["dataset"]["params"]["input_size"])));
            system_dict["local"]["transforms_val"].append(transforms.ToTensor());
            system_dict["local"]["applied_train_tensor"] = True;
    system_dict["local"]["data_transforms"]["train"] = transforms.Compose(system_dict["local"]["transforms_train"]);
    system_dict["local"]["data_transforms"]["val"] = transforms.Compose(system_dict["local"]["transforms_val"]);

    return system_dict;


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_transform_test(system_dict):
    if(not system_dict["local"]["normalize"]):
        if(not system_dict["local"]["applied_test_tensor"]):
            system_dict["local"]["transforms_test"].append(transforms.Resize(size=(system_dict["dataset"]["params"]["input_size"], system_dict["dataset"]["params"]["input_size"])));
            system_dict["local"]["transforms_test"].append(transforms.ToTensor());
            system_dict["local"]["applied_test_tensor"] = True;
    system_dict["local"]["data_transforms"]["test"] = transforms.Compose(system_dict["local"]["transforms_test"]);

    return system_dict;