from monk.pytorch.models.imports import *
from monk.system.imports import *
from monk.pytorch.models.models import *
from monk.pytorch.models.common import create_final_layer
from monk.pytorch.models.common import get_layer_uid
from monk.pytorch.models.layers import custom_model_get_layer
from monk.pytorch.models.layers import Net_Add
from monk.pytorch.models.layers import Net_Concat



@accepts(dict, path=[str, bool], final=bool, resume=bool, external_path=[bool, str, list], post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def load_model(system_dict, path=False, final=False, resume=False, external_path=False):
    '''
    Load model based on the system state 

    Args:
        system_dict (dict): System Dictionary
        path (str): Path to final or best model weights if Final flag is set
        final (bool): If True, Load model generated from latest epoch training
        resume (bool): If True, load model from last checkpoint to resume training
        external_path (str): Path to custom model weights

    Returns:
        network: Neural network loaded with weights.
    '''
    if(not torch.cuda.is_available()):
        if(final):
            if(path):
                finetune_net = torch.load(path + "final", map_location=torch.device('cpu'));
            else:
                finetune_net = torch.load(system_dict["model_dir_relative"] + "final", map_location=torch.device('cpu'));
        if(resume):
            finetune_net = torch.load(system_dict["model_dir_relative"] + "resume_state", map_location=torch.device('cpu'));
     
        if(external_path):
            finetune_net = torch.load(external_path, map_location=torch.device('cpu'));
    else:
        if(final):
            if(path):
                finetune_net = torch.load(path + "final");
            else:
                finetune_net = torch.load(system_dict["model_dir_relative"] + "final");
        if(resume):
            finetune_net = torch.load(system_dict["model_dir_relative"] + "resume_state");
     
        if(external_path):
            finetune_net = torch.load(external_path);

    return finetune_net;






@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def setup_model(system_dict):
    '''
    Setup model based on the system state and parameters

    Args:
        system_dict (dict): System Dictionary

    Returns:
        dict: Updated system dictionary
    '''
    if(system_dict["model"]["type"] == "pretrained"):

        model_name = system_dict["model"]["params"]["model_name"];
        use_pretrained = system_dict["model"]["params"]["use_pretrained"];
        freeze_base_network = system_dict["model"]["params"]["freeze_base_network"];
        custom_network = system_dict["model"]["custom_network"];
        final_layer = system_dict["model"]["final_layer"];
        num_classes = system_dict["dataset"]["params"]["num_classes"];

        finetune_net, model_name = get_base_model(model_name, use_pretrained, num_classes, freeze_base_network);


        if(len(custom_network)):
            if(final_layer):
                if(model_name in set1):
                    finetune_net = create_final_layer(finetune_net, custom_network, num_classes, set=1);
                elif(model_name in set2):
                    finetune_net = create_final_layer(finetune_net, custom_network, num_classes, set=2);
                elif(model_name in set3):
                    if(model_name == "inception_v3"):
                        msg = "Custm layer addition to inception_V3 unimplemented.\n";
                        msg += "Using basic inception_v3";
                        ConstraintWarning(msg);
                        num_ftrs = finetune_net.AuxLogits.fc.in_features;
                        finetune_net.AuxLogits.fc = nn.Linear(num_ftrs, num_classes);
                        num_ftrs = finetune_net.fc.in_features;
                        finetune_net.fc = nn.Linear(num_ftrs,num_classes);
                    else:
                        finetune_net = create_final_layer(finetune_net, custom_network, num_classes, set=3);
                elif(model_name in set4):
                    finetune_net = create_final_layer(finetune_net, custom_network, num_classes, set=4);
            else:
                msg = "Final layer not assigned";
                raise ConstraintError(msg);
        else:
            if(model_name in set1):
                num_ftrs = finetune_net.classifier[6].in_features;
                finetune_net.classifier[6] = nn.Linear(num_ftrs, num_classes);
            elif(model_name in set2):
                num_ftrs = finetune_net.classifier.in_features;
                finetune_net.classifier = nn.Linear(num_ftrs, num_classes);
            elif(model_name in set3):
                if(model_name == "inception_v3"):
                    num_ftrs = finetune_net.AuxLogits.fc.in_features;
                    finetune_net.AuxLogits.fc = nn.Linear(num_ftrs, num_classes);
                    num_ftrs = finetune_net.fc.in_features;
                    finetune_net.fc = nn.Linear(num_ftrs,num_classes);
                else:
                    num_ftrs = finetune_net.fc.in_features;
                    finetune_net.fc = nn.Linear(num_ftrs, num_classes);
            elif(model_name in set4):
                num_ftrs = finetune_net.classifier[1].in_features;
                finetune_net.classifier[1] = nn.Linear(num_ftrs, num_classes);



        system_dict["local"]["model"] = finetune_net;


        return system_dict;

    else:

        net = create_network(system_dict["custom_model"]["network_stack"], system_dict["dataset"]["params"]["data_shape"]);


        system_dict["local"]["model"] = net;

        return system_dict;








#@TraceFunction(trace_args=True, trace_rv=False)
def create_block(network_stack, count, G, sequential_first, position, current_width, current_in_shape):
    '''
    Recursively create sub-blocks when designing custom networks

    Args:
        network_stack (list): List of lists containing information on layers for the sub-branch in the network
        count (dict): A dictionary mapping to a count of every type of layer in the network
        G (directed graph): NetworkX object
        sequential_first (str): NAme of the current input layer
        position (int): Vertical position on the directed graph
        current_width (int): Horizontal position on the directed graph
        current_in_shape (tuple): Input shape to sub-network
    
    Returns:
        neural network: The required sub-branch
        directed graph: Updated directed graph
        str: Name of the outermost layer in the sub-network
        int: Vertical position of the outer most layer in the sub-network 
        int: Horizontal position of the outer most layer in the sub-network 
        tuple: Output shape from this sub-network
    '''
    position += 1;
    max_width = current_width
    net = nn.Sequential();
    for i in range(len(network_stack)): 
        if(type(network_stack[i]) == list):
            is_block = True;

            if(type(network_stack[i][-1]) != list):
                if(network_stack[i][-1]["name"] in ["add", "concatenate"]):
                    is_block=False;

            if(is_block):
                block, G, count, sequential_second, position, _, current_in_shape =  create_block(network_stack[i], count,
                                                                                        G, sequential_first, position, current_width, 
                                                                                        current_in_shape)
                #print(sequential_second, current_in_shape)
                sequential_first = sequential_second
                net.add_module(sequential_second, block)
            else:
                branch_end_points = [];
                branch_max_length = 0;
                branches = [];
                branch_net = [];

                #if(max_width < len(network_stack[i])-2):
                #    max_width = len(network_stack[i])-2;
                max_width = current_width;
                width = current_width;
                for j in range(len(network_stack[i])-1):
                    small_net = [];
                    branch_net.append(nn.Sequential())
                    branch_first = sequential_first
                    branch_position = position
                    column = max((j+1)*2+current_width, width);
                    max_width = column
                    current_in_shape_base = current_in_shape
                    for k in range(len(network_stack[i][j])):
                        if type(network_stack[i][j][k]) == list:
                            is_block2 = True;

                            if(type(network_stack[i][j][k][-1]) != list):
                                if(network_stack[i][j][k][-1]["name"] in ["add", "concatenate"]):
                                    is_block2=False;

                            if(is_block2):
                                block, G, count, branch_second, branch_position, width, current_in_shape_base = create_block(network_stack[i][j][k], 
                                                                                                                                 count,
                                                                                                                                 G, 
                                                                                                                                 branch_first, 
                                                                                                                                 branch_position, 
                                                                                                                                 column-2, #j+k+width
                                                                                                                                 current_in_shape_base) 
                            else:
                                block, G, count, branch_second, branch_position, width, current_in_shape_base = create_block([network_stack[i][j][k]], 
                                                                                                                                 count,
                                                                                                                                 G, 
                                                                                                                                 branch_first, 
                                                                                                                                 branch_position, 
                                                                                                                                 column-2, #j+k+width
                                                                                                                                 current_in_shape_base) 
                            #print(branch_second, current_in_shape_base)
                            branch_first = branch_second
                            small_net.append(block);
                            branch_net[j].add_module(branch_second, block);
                        else:
                            branch_second, count = get_layer_uid(network_stack[i][j][k], count);
                            current_layer, current_in_shape_base = custom_model_get_layer(network_stack[i][j][k], current_in_shape_base);
                            #print(branch_second, current_in_shape_base)
                            branch_net[j].add_module(branch_second, current_layer);
                            small_net.append(current_layer);
                            G.add_node(branch_second, pos=(column, branch_position));
                            branch_position += 1;
                            G.add_edge(branch_first, branch_second);
                            branch_first = branch_second;

                        branch_max_length = max(branch_position, branch_max_length)
                        if(k == len(network_stack[i][j])-1):
                            branch_end_points.append(branch_second);
                    branches.append(small_net);

                position = branch_max_length;
                position += 1;
                max_width += 2;

                sequential_second, count = get_layer_uid(network_stack[i][-1], count);
                if(network_stack[i][-1]["name"] == "concatenate"):
                    subnetwork = Net_Concat(branch_net);
                else:
                    subnetwork = Net_Add(branch_net);

                if(len(current_in_shape) == 2):
                    c, w = current_in_shape
                    x = torch.randn(1, c, w);
                    y = subnetwork(x)
                    current_in_shape = (y.shape[1], y.shape[2]);
                elif(len(current_in_shape) == 3):
                    c, h, w = current_in_shape
                    x = torch.randn(1, c, h, w);
                    y = subnetwork(x)
                    current_in_shape = (y.shape[1], y.shape[2], y.shape[3]);
                elif(len(current_in_shape) == 4):
                    c, d, h, w = current_in_shape
                    x = torch.randn(1, c, d, h, w);
                    y = subnetwork(x)
                    current_in_shape = (y.shape[1], y.shape[2], y.shape[3], y.shape[4]);

                #print(sequential_second, current_in_shape)

                G.add_node(sequential_second, pos=(2 + current_width, position));
                position += 1;
                for i in range(len(branch_end_points)):
                    G.add_edge(branch_end_points[i], sequential_second);
                sequential_first = sequential_second;
                net.add_module(sequential_second, subnetwork)
        else:
            sequential_second, count = get_layer_uid(network_stack[i], count);
            current_layer, current_in_shape = custom_model_get_layer(network_stack[i], current_in_shape);
            #print(sequential_second, current_in_shape)
            net.add_module(sequential_second, current_layer);
            G.add_node(sequential_second, pos=(2 + current_width, position))
            position += 1;
            G.add_edge(sequential_first, sequential_second);
            sequential_first = sequential_second;
            
    return net, G, count, sequential_second, position, max_width, current_in_shape
    


#@TraceFunction(trace_args=True, trace_rv=False)
def create_network(network_stack, current_in_shape):
    '''
    Main function to create network when designing custom networks

    Args:
        network_stack (list): List of lists containing information on layers in the network
    
    Returns:
        neural network: The required complete network
    '''
    count = [];
    for i in range(len(names)):
        count.append(1);

    G=nx.DiGraph()
    G.add_node("Net", pos=(1,1))
    sequential_first = "data";
    #sequential_second, count = get_layer_uid(network_stack[0], count);

    count = [];
    for i in range(len(names)):
        count.append(1);

    position = 1;
    G.add_node(sequential_first, pos=(2,1))
    position += 1;


    net = nn.Sequential();
    max_width = 1;
    width = 0;
    for i in range(len(network_stack)):
        if(type(network_stack[i]) == list):
            is_block = True;

            if(type(network_stack[i][-1]) != list):
                if(network_stack[i][-1]["name"] in ["add", "concatenate"]):
                    is_block=False;


            if(is_block):
                block, G, count, sequential_second, position, _, current_in_shape =  create_block(network_stack[i], count,
                                                                                        G, sequential_first, position, 0,
                                                                                        current_in_shape)
                #print(sequential_second, current_in_shape)
                sequential_first = sequential_second
                net.add_module(sequential_second, block)
            else:
                branch_end_points = [];
                branch_max_length = 0;
                branches = [];
                branch_net = [];


                if(max_width < len(network_stack[i])-2):
                    max_width = len(network_stack[i])-2;
                width = 0;
                for j in range(len(network_stack[i])-1):
                    small_net = [];
                    branch_first = sequential_first
                    branch_net.append(nn.Sequential())
                    branch_position = position
                    current_in_shape_base = current_in_shape
                    if(width > 0):
                        if(column == width):
                            column += 2;
                        else:
                            column = width;
                    else:
                        column = (j+1)*2;
                    for k in range(len(network_stack[i][j])):
                        if type(network_stack[i][j][k]) == list:
                            is_block2 = True;

                            if(type(network_stack[i][j][k][-1]) != list):
                                if(network_stack[i][j][k][-1]["name"] in ["add", "concatenate"]):
                                    is_block2=False;
                        
                            if(is_block2):
                                block, G, count, branch_second, branch_position, width, current_in_shape_base = create_block(network_stack[i][j][k], 
                                                                                                             count,
                                                                                                             G, 
                                                                                                             branch_first, 
                                                                                                             branch_position, 
                                                                                                             column-2, #j+k+width
                                                                                                             current_in_shape_base)
                            else:
                                block, G, count, branch_second, branch_position, width, current_in_shape_base = create_block([network_stack[i][j][k]], 
                                                                                                             count,
                                                                                                             G, 
                                                                                                             branch_first, 
                                                                                                             branch_position, 
                                                                                                             column-2,
                                                                                                             current_in_shape_base)
                            #print(branch_second, current_in_shape_base)
                            branch_first = branch_second
                            small_net.append(block);
                            branch_net[j].add_module(branch_second, block);
                        else:
                            branch_second, count = get_layer_uid(network_stack[i][j][k], count);
                            current_layer, current_in_shape_base = custom_model_get_layer(network_stack[i][j][k], current_in_shape_base);
                            #print(branch_second, current_in_shape_base)
                            branch_net[j].add_module(branch_second, current_layer);
                            small_net.append(current_layer);
                            G.add_node(branch_second, pos=(column, branch_position));
                            branch_position += 1;
                            G.add_edge(branch_first, branch_second);
                            branch_first = branch_second;                        

                        branch_max_length = max(branch_position, branch_max_length)
                        if(k == len(network_stack[i][j])-1):
                            branch_end_points.append(branch_second);
                    branches.append(small_net);
                
                position = branch_max_length;
                position += 1;
                max_width += width

                sequential_second, count = get_layer_uid(network_stack[i][-1], count)
                if(network_stack[i][-1]["name"] == "concatenate"):
                    subnetwork = Net_Concat(branch_net);
                else:
                    subnetwork = Net_Add(branch_net);

                if(len(current_in_shape) == 2):
                    c, w = current_in_shape
                    x = torch.randn(1, c, w);
                    y = subnetwork(x)
                    current_in_shape = (y.shape[1], y.shape[2]);
                elif(len(current_in_shape) == 3):
                    c, h, w = current_in_shape
                    x = torch.randn(1, c, h, w);
                    y = subnetwork(x)
                    current_in_shape = (y.shape[1], y.shape[2], y.shape[3]);
                elif(len(current_in_shape) == 4):
                    c, d, h, w = current_in_shape
                    x = torch.randn(1, c, d, h, w);
                    y = subnetwork(x)
                    current_in_shape = (y.shape[1], y.shape[2], y.shape[3], y.shape[4]);

                #print(sequential_second, current_in_shape)

                sequential_second, count = get_layer_uid(network_stack[i][-1], count);

                G.add_node(sequential_second, pos=(2, position));
                position += 1;
                for i in range(len(branch_end_points)):
                    G.add_edge(branch_end_points[i], sequential_second);
                sequential_first = sequential_second;
                net.add_module(sequential_second, subnetwork)


        else:
            sequential_second, count = get_layer_uid(network_stack[i], count);
            G.add_node(sequential_second, pos=(2, position))
            current_layer, current_in_shape = custom_model_get_layer(network_stack[i], current_in_shape);
            #print(sequential_second, current_in_shape)
            net.add_module(sequential_second, current_layer);
            position += 1;
            G.add_edge(sequential_first, sequential_second);
            sequential_first = sequential_second;


    max_width = max(max_width, width);
    if(max_width == 1):
        G.add_node("monk", pos=(3, position));
    else:
        G.add_node("monk", pos=(max_width + 3, position))
    pos = nx.get_node_attributes(G, 'pos')


    plt.figure(3, figsize=(12, 12 + position//6)) 
    nx.draw_networkx(G, pos, with_label=True, font_size=16, node_color="yellow", node_size=100)
    plt.savefig("graph.png");

    return net;





#@TraceFunction(trace_args=True, trace_rv=False)
def debug_create_block(network_stack, count, G, sequential_first, position, current_width):
    '''
    Recursively visualize sub-blocks when designing custom networks

    Args:
        network_stack (list): List of lists containing information on layers for the sub-branch in the network
        count (dict): A dictionary mapping to a count of every type of layer in the network
        G (directed graph): NetworkX object
        sequential_first (str): NAme of the current input layer
        position (int): Vertical position on the directed graph
        current_width (int): Horizontal position on the directed graph
        current_in_shape (tuple): Input shape to sub-network
    
    Returns:
        neural network: The required sub-branch
        directed graph: Updated directed graph
        str: Name of the outermost layer in the sub-network
        int: Vertical position of the outer most layer in the sub-network 
        int: Horizontal position of the outer most layer in the sub-network 
        tuple: Output shape from this sub-network
    '''
    position += 1;
    max_width = current_width
    for i in range(len(network_stack)): 
        if(type(network_stack[i]) == list):
            is_block = True;

            if(type(network_stack[i][-1]) != list):
                if(network_stack[i][-1]["name"] in ["add", "concatenate"]):
                    is_block=False;

            if(is_block):
                G, count, sequential_second, position, _ =  debug_create_block(network_stack[i], count,
                                                                                G, sequential_first, position, current_width) #0
                sequential_first = sequential_second
            else:
                branch_end_points = [];
                branch_max_length = 0;
                branches = [];
                branch_net = [];

                #if(max_width < len(network_stack[i])-2):
                #    max_width = len(network_stack[i])-2;
                max_width = current_width;
                width = current_width;
                for j in range(len(network_stack[i])-1):
                    branch_first = sequential_first
                    branch_position = position
                    column = max((j+1)*2+current_width, width);
                    max_width = column
                    for k in range(len(network_stack[i][j])):
                        if type(network_stack[i][j][k]) == list:
                            is_block2 = True;

                            if(type(network_stack[i][j][k][-1]) != list):
                                if(network_stack[i][j][k][-1]["name"] in ["add", "concatenate"]):
                                    is_block2=False;


                            if(is_block2):
                                G, count, branch_second, branch_position, width = debug_create_block(network_stack[i][j][k], 
                                                                                     count,
                                                                                     G, 
                                                                                     branch_first, 
                                                                                     branch_position, 
                                                                                     column-2) #j+k+width, j*2+current_width
                            else:
                                G, count, branch_second, branch_position, width = debug_create_block([network_stack[i][j][k]], 
                                                                                     count,
                                                                                     G, 
                                                                                     branch_first, 
                                                                                     branch_position, 
                                                                                     column-2) #j+k+width, j+k+current_width
                            branch_first = branch_second
                        else:
                            branch_second, count = get_layer_uid(network_stack[i][j][k], count);
                            G.add_node(branch_second, pos=(column, branch_position));
                            branch_position += 1;
                            G.add_edge(branch_first, branch_second);
                            branch_first = branch_second;

                        branch_max_length = max(branch_position, branch_max_length)
                        if(k == len(network_stack[i][j])-1):
                            branch_end_points.append(branch_second);
                
                position = branch_max_length;
                position += 1;
                max_width += 2;

                sequential_second, count = get_layer_uid(network_stack[i][-1], count);

                G.add_node(sequential_second, pos=(2 + current_width, position));
                position += 1;
                for i in range(len(branch_end_points)):
                    G.add_edge(branch_end_points[i], sequential_second);
                sequential_first = sequential_second;
        else:
            sequential_second, count = get_layer_uid(network_stack[i], count);
            G.add_node(sequential_second, pos=(2+current_width, position))
            position += 1;
            G.add_edge(sequential_first, sequential_second);
            sequential_first = sequential_second;
    
    return G, count, sequential_second, position, max_width
    

#@TraceFunction(trace_args=True, trace_rv=False)
def debug_create_network(network_stack):
    '''
    Main function to visualize network when designing custom networks

    Args:
        network_stack (list): List of lists containing information on layers in the network
    
    Returns:
        neural network: The required complete network
    '''
    count = [];
    for i in range(len(names)):
        count.append(1);

    G=nx.DiGraph()
    G.add_node("Net", pos=(1,1))
    sequential_first = "data";
    #sequential_second, count = get_layer_uid(network_stack[0], count);

    count = [];
    for i in range(len(names)):
        count.append(1);

    position = 1;
    G.add_node(sequential_first, pos=(2,1))
    position += 1;


    max_width = 1;
    width = 0;
    for i in range(len(network_stack)):
        if(type(network_stack[i]) == list):
            is_block = True;

            if(type(network_stack[i][-1]) != list):
                if(network_stack[i][-1]["name"] in ["add", "concatenate"]):
                    is_block=False;

            if(is_block):
                G, count, sequential_second, position, _ =  debug_create_block(network_stack[i], count,
                                                                                G, sequential_first, position, 0)
                sequential_first = sequential_second
            else:
                branch_end_points = [];
                branch_max_length = 0;
                branches = [];
                branch_net = [];


                if(max_width < len(network_stack[i])-2):
                    max_width = len(network_stack[i])-2;
                width = 0;
                for j in range(len(network_stack[i])-1):
                    branch_first = sequential_first
                    branch_position = position
                    if(width > 0):
                        if(column == width):
                            column += 2;
                        else:
                            column = width;
                    else:
                        column = (j+1)*2;
                    for k in range(len(network_stack[i][j])):
                        if type(network_stack[i][j][k]) == list:
                            is_block2 = True;

                            if(type(network_stack[i][j][k][-1]) != list):
                                if(network_stack[i][j][k][-1]["name"] in ["add", "concatenate"]):
                                    is_block2=False;

                           

                            if(is_block2):
                                G, count, branch_second, branch_position, width = debug_create_block(network_stack[i][j][k], 
                                                                                     count,
                                                                                     G, 
                                                                                     branch_first, 
                                                                                     branch_position, 
                                                                                     column-2) #j*2+width
                            else:
                                G, count, branch_second, branch_position, width = debug_create_block([network_stack[i][j][k]], 
                                                                                     count,
                                                                                     G, 
                                                                                     branch_first, 
                                                                                     branch_position, 
                                                                                     column-2) #j+k+width
                            branch_first = branch_second
                        else:
                            branch_second, count = get_layer_uid(network_stack[i][j][k], count);
                            G.add_node(branch_second, pos=(column, branch_position));
                            branch_position += 1;
                            G.add_edge(branch_first, branch_second);
                            branch_first = branch_second;

                        branch_max_length = max(branch_position, branch_max_length)
                        if(k == len(network_stack[i][j])-1):
                            branch_end_points.append(branch_second);
                    
                position = branch_max_length;
                position += 1;
                max_width += width

                sequential_second, count = get_layer_uid(network_stack[i][-1], count);

                G.add_node(sequential_second, pos=(2, position));
                position += 1;
                for i in range(len(branch_end_points)):
                    G.add_edge(branch_end_points[i], sequential_second);
                sequential_first = sequential_second;


        else:
            sequential_second, count = get_layer_uid(network_stack[i], count);
            G.add_node(sequential_second, pos=(2, position))
            position += 1;
            G.add_edge(sequential_first, sequential_second);
            sequential_first = sequential_second;

    max_width = max(max_width, width);
    if(max_width == 1):
        G.add_node("monk", pos=(3, position));
    else:
        G.add_node("monk", pos=(max_width + 3, position))
    pos = nx.get_node_attributes(G, 'pos')


    plt.figure(3, figsize=(16, 20 + position//6)) 
    nx.draw_networkx(G, pos, with_label=True, font_size=16, node_color="yellow", node_size=100)
    plt.savefig("graph.png");