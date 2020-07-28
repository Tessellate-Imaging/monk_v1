from monk.tf_keras_1.models.imports import *
from monk.system.imports import *
from monk.tf_keras_1.models.layers import get_layer



@accepts("self", bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def set_parameter_requires_grad(finetune_net, freeze_base_network):
    '''
    Freeze based network as per params set

    Args:
        finetune_net (network): Model network
        freeze_base_network (bool): If True, all trainable params are freezed

    Returns:
        network: Updated Model network
    '''
    if freeze_base_network:
        for layer in finetune_net.layers:
            layer.trainable=False
    else:
        for layer in finetune_net.layers:
            layer.trainable=True

    return finetune_net



@accepts(list, "self", post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def set_final_layer(custom_network, x):
    '''
    Setup final sub-network 

    Args:
        custom_network (list): List of dicts containing details on appeded layers to base netwoek in transfer learning
        num_ftrs (int): Number of features coming from base network's last layers
        num_classes (int): Number of classes in the dataset

    Returns:
        layer: Sequential sub-network with added layers 
    '''
    for i in range(len(custom_network)):
        x = get_layer(custom_network[i], x);
    return x;





@accepts("self", list, int, set=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def create_final_layer(finetune_net, custom_network, num_classes):
    '''
    Create final sub-network 

    Args:
        finetune_net (network): Initial base network
        custom_network (list):  List of dicts containing details on appeded layers to base netwoek in transfer learning
        num_classes (int): Number of classes in the dataset

    Returns:
        network: Updated base network with appended custom additions
    '''
    x = finetune_net.output
    x = krl.GlobalAveragePooling2D()(x);

    preds = set_final_layer(custom_network, x);

    finetune_net = keras.models.Model(inputs=finetune_net.input, outputs=preds);
    return finetune_net;





@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def get_num_layers(system_dict):
    '''
    Depricated function - Get number of potentially trainable layers

    Args:
        system_dict (dict): System dict containing system state and parameters

    Returns:
        dict: Updated system dict 
    '''
    complete_list = [];
    for layer in system_dict["local"]["model"].layers:
        if(layer.count_params() > 0):
            complete_list.append(layer.name)
    system_dict["model"]["params"]["num_layers"] = len(complete_list);
    return system_dict;



@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def get_num_trainable_layers(system_dict):
    '''
    Get number of potentially trainable layers

    Args:
        system_dict (dict): System dict containing system state and parameters

    Returns:
        dict: Updated system dict 
    '''
    complete_list_trainable = [];
    for layer in system_dict["local"]["model"].layers:
        if(layer.count_params() > 0):
            if(layer.trainable):
                complete_list_trainable.append(layer.name)
    system_dict["model"]["params"]["num_params_to_update"] = len(complete_list_trainable);
    return system_dict;




@accepts(int, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def freeze_layers(num, system_dict):
    '''
    Main function responsible to freeze layers in network

    Args:
        num (int): Number of layers to freeze
        system_dict (dict): System dict containing system state and parameters

    Returns:
        dict: Updated system dict 
    '''
    system_dict = get_num_layers(system_dict);
    num_layers_in_model = system_dict["model"]["params"]["num_layers"];
    if(num > num_layers_in_model):
        msg = "Parameter num > num_layers_in_model\n";
        msg += "Freezing entire network\n";
        msg += "TIP: Total layers: {}".format(num_layers_in_model);
        raise ConstraintError(msg);

    current_num = 0;
    value = False;
    for layer in system_dict["local"]["model"].layers:
        if(layer.count_params() > 0):
            layer.trainable = value;
            current_num += 1;
            if(current_num == num):
                value = True;   

    system_dict = get_num_trainable_layers(system_dict);
    system_dict["model"]["status"] = True;   

    return system_dict;



@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def print_grad_stats(system_dict):
    '''
    Print details on which layers are trainable

    Args:
        system_dict (dict): System dict containing system state and parameters

    Returns:
        None 
    '''
    print("Model - Gradient Statistics");
    i = 1;
    for layer in system_dict["local"]["model"].layers:
        print("    {}. {} Trainable - {}".format(i+1, layer.name, layer.trainable));
        i += 1;
    print("");





@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def setup_device_environment(system_dict):
    '''
    Load model weights on device - cpu or gpu 

    Args:
        system_dict (dict): System dict containing system state and parameters

    Returns:
        dict: Updated system dict 
    '''
    num_cores = psutil.cpu_count();
    if system_dict["model"]["params"]["use_gpu"]:
        num_GPU = 1
        num_CPU = 1
    else:
        num_CPU = 1
        num_GPU = 0

    if(tf.__version__.split(".")[0] == "2"):
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=system_dict["model"]["params"]["gpu_memory_fraction"], 
            allow_growth = True)
        
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores, 
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True, 
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU}, 
            gpu_options=gpu_options)
        
        session = tf.compat.v1.Session(config=config)
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=system_dict["model"]["params"]["gpu_memory_fraction"], 
            allow_growth = True)
        
        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, 
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True, 
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU}, 
            gpu_options=gpu_options)
        
        session = tf.Session(config=config)

    K.set_session(session);
    
    return system_dict;




@accepts(dict, list, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def get_layer_uid(network_stack, count):
    '''
    Get a unique name for layer in custom network development

    Args:
        network_stack (list): List of list containing custom network details
        count (dict): a unique dictionary mapping number of every type of layer in the network
        system_dict (dict): System dict containing system state and parameters

    Returns:
        str: layer unique name
        dict: updated layer type mapper count
    '''
    if network_stack["uid"]:
        return network_stack["uid"], count;
    else:
        index = layer_names.index(network_stack["name"]);
        network_name = names[index] + str(count[index]);
        count[index] += 1;
        return network_name, count;
    





        
