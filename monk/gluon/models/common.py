from gluon.models.imports import *
from system.imports import *
from gluon.models.layers import get_layer


@accepts("self", bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_parameter_requires_grad(finetune_net, freeze_base_network):
    if(freeze_base_network):
        for param in finetune_net.collect_params().values():
            param.grad_req = 'null';
    return finetune_net


@accepts(dict, activation=str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=True)
def get_final_layer(network_layer, activation='relu'):
    act_list = ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign'];

    if(activation not in act_list):
        print("Final activation must be in set: {}".format(act_list));
        print("");
    else:
        layer = nn.Dense(network_layer["params"]["out_features"], weight_initializer=init.Xavier(), activation=activation);
        return layer;


@accepts("self", list, int, set=int, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def create_final_layer(finetune_net, custom_network, num_classes, set=1):
    last_layer_name = custom_network[len(custom_network)-1]["name"];

    if(set==1):
        if(last_layer_name == "linear"):
            with finetune_net.name_scope():
                for i in range(len(custom_network)-1):
                    layer = get_layer(custom_network[i]);
                    finetune_net.features.add(layer);
                    finetune_net.features[len(finetune_net.features)-1].initialize(init.Xavier(), ctx = ctx);
                finetune_net.output = get_layer(custom_network[len(custom_network)-1])
                finetune_net.output.initialize(init.Xavier(), ctx = ctx)
        else:
            with finetune_net.name_scope():
                for i in range(len(custom_network)-2):
                    layer = get_layer(custom_network[i]);
                    finetune_net.features.add(layer);
                    finetune_net.features[len(finetune_net.features)-1].initialize(init.Xavier(), ctx = ctx);
                finetune_net.output = get_final_layer(custom_network[len(custom_network)-2], activation=custom_network[len(custom_network)-1]['name']);
                finetune_net.output.initialize(init.Xavier(), ctx = ctx)


    if(set==2):
        net = nn.HybridSequential();
        with net.name_scope():
            for i in range(len(custom_network)):
                layer = get_layer(custom_network[i]);
                net.add(layer);
        with finetune_net.name_scope():
            finetune_net.output = net; 
            finetune_net.output.initialize(init.Xavier(), ctx = ctx) 


    if(set==3):
        msg = "Custom model addition for - Set 3 models: Not Implemented.\n";
        msg += "Set 3 models - {}\n".format(set3);
        msg += "Ignoring added layers\n";
        ConstraintWarning(msg);

    return finetune_net;



@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def model_to_device(system_dict):
    GPUs = GPUtil.getGPUs()
    if(len(GPUs)==0):
        system_dict["local"]["ctx"] = [mx.cpu()];
    else:
        if(system_dict["model"]["params"]["use_gpu"]):
            system_dict["local"]["ctx"] = [mx.gpu(0)];
        else:
            system_dict["local"]["ctx"] = [mx.cpu()];

    system_dict["local"]["model"].collect_params().reset_ctx(system_dict["local"]["ctx"])
    system_dict["local"]["model"].hybridize()

    return system_dict;


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def print_grad_stats(system_dict):
    print("Model - Gradient Statistics");
    i = 1;
    for param in system_dict["local"]["model"].collect_params().values():
        if(i%2 != 0):
            print("    {}. {} Trainable - {}".format(i//2+1, param, param.grad_req ));
        i += 1;
    print("");


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def get_num_layers(system_dict):
    if(system_dict["model"]["type"] == "pretrained"):
        num_layers = 0;
        complete_list = [];
        for names in system_dict["local"]["model"].collect_params().keys():
            layer_name = '_'.join(names.split("_")[:-1]);
            if(layer_name not in complete_list):
                complete_list.append(layer_name);
        system_dict["model"]["params"]["num_layers"] = len(np.unique(complete_list));
        return system_dict;
    else:
        num_layers = 0;
        complete_list = [];
        for names in system_dict["local"]["model"].collect_params().keys():
            complete_list.append(names.split("_")[0]);
        system_dict["model"]["params"]["num_layers"] = len(np.unique(complete_list));
        return system_dict;



@accepts(int, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def freeze_layers(num, system_dict):
    if(system_dict["model"]["type"] == "pretrained"):
        num_freeze = num;
        num_freezed = 0;
        training_list = system_dict["local"]["params_to_update"];

        grad_req= "null";
        current_name = training_list[0];
        for param in system_dict["local"]["model"].collect_params().values():
            if(current_name != '_'.join(param.name.split("_")[:-1])):
                num_freezed += 1;
                if(num_freezed == num_freeze):
                    grad_req = "write";
                param.grad_req = grad_req;
                current_name = '_'.join(param.name.split("_")[:-1]);
            else:
                param.grad_req = grad_req;

        current_name = "";
        ip = 0;
        system_dict["local"]["params_to_update"] = [];
        for param in system_dict["local"]["model"].collect_params().values():
            if(ip==0):
                current_name = '_'.join(param.name.split("_")[:-1]);
                if(param.grad_req == "write"):
                    system_dict["local"]["params_to_update"].append(current_name);
            else:
                if(current_name != '_'.join(param.name.split("_")[:-1])):
                    current_name = '_'.join(param.name.split("_")[:-1]);
                    if(param.grad_req == "write"):
                        system_dict["local"]["params_to_update"].append(current_name);
            ip += 1;
        system_dict["model"]["params"]["num_params_to_update"] = len(system_dict["local"]["params_to_update"]);
        system_dict["model"]["status"] = True;

        return system_dict;

    else:
        num_freeze = num;
        num_freezed = 0;
        training_list = system_dict["local"]["params_to_update"];


        current_name = training_list[0];
        for param in system_dict["local"]["model"].collect_params().values():
            if(current_name != param.name.split("_")[0]):
                num_freezed += 1;
                if(num_freezed == num_freeze):
                    break;
                param.grad_req = "null";
                current_name = param.name.split("_")[0];
            else:
                param.grad_req = "null";


        current_name = "";
        ip = 0;
        system_dict["local"]["params_to_update"] = [];
        for param in system_dict["local"]["model"].collect_params().values():
            if(ip==0):
                current_name = param.name.split("_")[0];
                if(param.grad_req == "write"):
                    system_dict["local"]["params_to_update"].append(current_name);
            else:
                if(current_name != param.name.split("_")[0]):
                    current_name = param.name.split("_")[0];
                    if(param.grad_req == "write"):
                        system_dict["local"]["params_to_update"].append(current_name);
            ip += 1;
        system_dict["model"]["params"]["num_params_to_update"] = len(system_dict["local"]["params_to_update"]);
        system_dict["model"]["status"] = True;



        return system_dict;


@accepts(dict, list, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def get_layer_uid(network_stack, count):
    if network_stack["uid"]:
        return network_stack["uid"], count;
    else:
        index = layer_names.index(network_stack["name"]);
        network_name = names[index] + str(count[index]);
        count[index] += 1;
        return network_name, count;
