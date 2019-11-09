from pytorch.models.imports import *
from system.imports import *
from pytorch.models.layers import get_layer


@accepts("self", bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_parameter_requires_grad(finetune_net, freeze_base_network):
    if freeze_base_network:
        for param in finetune_net.parameters():
            param.requires_grad = False
    else:
        for param in finetune_net.parameters():
            param.requires_grad = True
    return finetune_net


@accepts(list, int, int, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def set_final_layer(custom_network, num_ftrs, num_classes):
    modules = [];
    for i in range(len(custom_network)):
        layer, num_ftrs = get_layer(custom_network[i], num_ftrs);
        modules.append(layer);
    sequential = nn.Sequential(*modules)
    return sequential;


@accepts("self", list, int, set=int, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def create_final_layer(finetune_net, custom_network, num_classes, set=1):
    if(set == 1):
        num_ftrs = finetune_net.classifier[6].in_features;
        finetune_net.classifier = set_final_layer(custom_network, num_ftrs, num_classes);
    elif(set == 2):
        num_ftrs = finetune_net.classifier.in_features;
        finetune_net.classifier = set_final_layer(custom_network, num_ftrs, num_classes);
    elif(set == 3):
        num_ftrs = finetune_net.fc.in_features;
        finetune_net.fc = set_final_layer(custom_network, num_ftrs, num_classes);
    elif(set == 4):
        num_ftrs = finetune_net.classifier[1].in_features;
        finetune_net.classifier = set_final_layer(custom_network, num_ftrs, num_classes);
    
    return finetune_net;



@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def model_to_device(system_dict):
    if(system_dict["model"]["params"]["use_gpu"]):
        system_dict["local"]["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        if(torch.cuda.is_available()):
            use_gpu = True;
            system_dict["model"]["params"]["use_gpu"] = use_gpu;
        else:
            use_gpu = False;
            system_dict["model"]["params"]["use_gpu"] = use_gpu;
    else:
        system_dict["local"]["device"] = torch.device("cpu");
    
    system_dict["local"]["model"] = system_dict["local"]["model"].to(system_dict["local"]["device"]);
    return system_dict;


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def print_grad_stats(system_dict):
    print("Model - Gradient Statistics");
    i = 1;
    for name, param in system_dict["local"]["model"].named_parameters():
        if(i%2 != 0):
            print("    {}. {} Trainable - {}".format(i//2+1, name, param.requires_grad ));
        i += 1;
    print("");



@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def get_num_layers(system_dict):
    num_layers = 0;
    for param in system_dict["local"]["model"].named_parameters():
        num_layers += 1;
    system_dict["model"]["params"]["num_layers"] = num_layers//2;
    return system_dict;



@accepts(int, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def freeze_layers(num, system_dict):
    system_dict = get_num_layers(system_dict);
    num_layers_in_model = system_dict["model"]["params"]["num_layers"];
    if(num > num_layers_in_model):
        msg = "Parameter num > num_layers_in_model\n";
        msg += "Freezing entire network\n";
        msg += "TIP: Total layers: {}".format(num_layers_in_model);
        raise ConstraintError(msg)

    num = num*2;
    current_num = 0;
    value = False;

    for name,param in system_dict["local"]["model"].named_parameters():
        param.requires_grad = value;
        current_num += 1;
        if(current_num == num):
            value = True;

    system_dict["local"]["params_to_update"] = []
    for name,param in system_dict["local"]["model"].named_parameters():
        if param.requires_grad == True:
            system_dict["local"]["params_to_update"].append(param);

    system_dict["model"]["params"]["num_params_to_update"] = len(system_dict["local"]["params_to_update"])//2;
    system_dict["model"]["status"] = True;

    return system_dict;