from tf_keras_1.models.imports import *
from system.imports import *
from tf_keras_1.models.models import *
from tf_keras_1.models.common import create_final_layer
from tf_keras_1.models.common import get_layer_uid
from tf_keras_1.models.layers import custom_model_get_layer



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
    if(final):
        if(path):
            finetune_net = keras.models.load_model(path + "final.h5");
        else:
            finetune_net = keras.models.load_model(system_dict["model_dir_relative"] + "final.h5");
    if(resume):
        finetune_net = keras.models.load_model(system_dict["model_dir_relative"] + "resume_state.h5");
 
    if(external_path):
        finetune_net = keras.models.load_model(external_path);

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
        input_size = system_dict["dataset"]["params"]["input_size"];

        finetune_net, model_name = get_base_model(model_name, use_pretrained, num_classes, freeze_base_network, input_size);


        if(len(custom_network)):
            if(final_layer):
                finetune_net = create_final_layer(finetune_net, custom_network, num_classes);
            else:
                msg = "Final layer not assigned";
                raise ConstraintError(msg);
        else:
            x = finetune_net.output
            x = krl.GlobalAveragePooling2D()(x)
            x = krl.Dense(512)(x);
            x = krl.ReLU()(x)
            x = krl.Dropout(0.5)(x);
            if(system_dict["dataset"]["label_type"] == "single" or system_dict["dataset"]["label_type"] == False):
                x = krl.Dense(num_classes)(x);
                preds = krl.Softmax()(x);
            else:
                x = krl.Dense(num_classes)(x);
                preds = krl.Activation('sigmoid')(x);
            finetune_net = keras.models.Model(inputs=finetune_net.input, outputs=preds);

        system_dict["local"]["model"] = finetune_net;

        return system_dict;


    else:
        net = create_network(system_dict["custom_model"]["network_stack"], 
            system_dict["dataset"]["params"]["data_shape"], 
            system_dict["custom_model"]["network_initializer"]);

        system_dict["local"]["model"] = net;

        return system_dict;





#@TraceFunction(trace_args=True, trace_rv=False)
def create_block(network_stack, count, G, sequential_first, position, current_width, out, initializer):
    '''
    Recursively create sub-blocks when designing custom networks

    Args:
        network_stack (list): List of lists containing information on layers for the sub-branch in the network
        count (dict): A dictionary mapping to a count of every type of layer in the network
        G (directed graph): NetworkX object
        sequential_first (str): NAme of the current input layer
        position (int): Vertical position on the directed graph
        current_width (int): Horizontal position on the directed graph
        out (placeholder): Keras placeholder temporary input for this sub-network
        initializer (keras iniatialzer): Random wieght initializer for layers in the sub-network
    
    Returns:
        neural network: The required sub-branch
        directed graph: Updated directed graph
        str: Name of the outermost layer in the sub-network
        int: Vertical position of the outer most layer in the sub-network 
        int: Horizontal position of the outer most layer in the sub-network 
    '''
    position += 1;
    max_width = current_width;
    for i in range(len(network_stack)): 
        if(type(network_stack[i]) == list):
            is_block = True;

            if(type(network_stack[i][-1]) != list):
                if(network_stack[i][-1]["name"] in ["add", "concatenate"]):
                    is_block=False;

            if(is_block):
                G, count, sequential_second, position, _, out, initializer =  create_block(network_stack[i], count,
                                                                                        G, sequential_first, position, 0,
                                                                                        out, initializer)
                #print(sequential_second, current_in_shape)
                sequential_first = sequential_second
            else:
                branch_end_points = [];
                branch_max_length = 0;
                branches = [];
                branch_net = [];
                out_base = out;

                #if(max_width < len(network_stack[i])-2):
                #    max_width = len(network_stack[i])-2;
                max_width = current_width;
                width = current_width;
                for j in range(len(network_stack[i])-1):
                    small_net = [];
                    branch_net.append(out_base)
                    branch_first = sequential_first
                    branch_position = position
                    column = max((j+1)*2+current_width, width);
                    max_width = column;
                    for k in range(len(network_stack[i][j])):
                        if type(network_stack[i][j][k]) == list:
                            is_block2 = True;

                            if(type(network_stack[i][j][k][-1]) != list):
                                if(network_stack[i][j][k][-1]["name"] in ["add", "concatenate"]):
                                    is_block2=False;

                            if(is_block2):
                                G, count, branch_second, branch_position, width, branch_net[j], initializer = create_block(network_stack[i][j][k], 
                                                                                                             count,
                                                                                                             G, 
                                                                                                             branch_first, 
                                                                                                             branch_position, 
                                                                                                             column-2, #j+k+width
                                                                                                             branch_net[j], 
                                                                                                             initializer)
                            else:
                                G, count, branch_second, branch_position, width, branch_net[j], initializer = create_block([network_stack[i][j][k]], 
                                                                                                             count,
                                                                                                             G, 
                                                                                                             branch_first, 
                                                                                                             branch_position, 
                                                                                                             column-2,
                                                                                                             branch_net[j],
                                                                                                             initializer)
                            #print(branch_second, current_in_shape_base)
                            branch_first = branch_second
                            small_net.append(branch_net[j]);
                        else:
                            branch_second, count = get_layer_uid(network_stack[i][j][k], count);
                            branch_net[j] = custom_model_get_layer(network_stack[i][j][k], initializer)(branch_net[j]);
                            small_net.append(custom_model_get_layer(network_stack[i][j][k], initializer));
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
                    out = keras.layers.concatenate(branch_net)
                else:
                    out = keras.layers.add(branch_net)

                G.add_node(sequential_second, pos=(2 + current_width, position));
                position += 1;
                for i in range(len(branch_end_points)):
                    G.add_edge(branch_end_points[i], sequential_second);
                sequential_first = sequential_second;
        else:
            sequential_second, count = get_layer_uid(network_stack[i], count);
            out = custom_model_get_layer(network_stack[i], initializer)(out);
            G.add_node(sequential_second, pos=(2 + current_width, position))
            position += 1;
            G.add_edge(sequential_first, sequential_second);
            sequential_first = sequential_second;
            
    return G, count, sequential_second, position, max_width, out, initializer
    


#@TraceFunction(trace_args=True, trace_rv=False)
def create_network(network_stack, current_in_shape, initializer):
    '''
    Main function to create network when designing custom networks

    Args:
        network_stack (list): List of lists containing information on layers in the network
        initializer (keras iniatialzer): Random wieght initializer for layers in the sub-network
    
    Returns:
        neural network: The required complete network
    '''
    count = [];
    for i in range(len(names)):
        count.append(1);

    G=nx.DiGraph()
    G.add_node("Net", pos=(1,1))
    sequential_first = "data";
    

    if(len(current_in_shape) == 2):
        c, w = current_in_shape;
        input1 = krl.Input(shape=(w, c));
    elif(len(current_in_shape) == 3):
        c, h, w = current_in_shape;
        input1 = krl.Input(shape=(h, w, c));
    elif(len(current_in_shape) == 4):
        c, d, h, w = current_in_shape;
        input1 = krl.Input(shape=(d, h, w, c));

    out = input1;

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
                G, count, sequential_second, position, _, out, initializer =  create_block(network_stack[i], count,
                                                                                        G, sequential_first, position, 0,
                                                                                        out, initializer)
                #print(sequential_second, current_in_shape)
                sequential_first = sequential_second
            else:
                branch_end_points = [];
                branch_max_length = 0;
                branches = [];
                branch_net = [];
                out_base = out;


                if(max_width < len(network_stack[i])-2):
                    max_width = len(network_stack[i])-2;
                width = 0;
                for j in range(len(network_stack[i])-1):
                    small_net = [];
                    branch_first = sequential_first
                    branch_net.append(out_base)
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
                                G, count, branch_second, branch_position, width, branch_net[j], initializer = create_block(network_stack[i][j][k], 
                                                                                                             count,
                                                                                                             G, 
                                                                                                             branch_first, 
                                                                                                             branch_position, 
                                                                                                             column-2, #j+k+width
                                                                                                             branch_net[j],
                                                                                                             initializer)
                            else:
                                G, count, branch_second, branch_position, width, branch_net[j], initializer = create_block([network_stack[i][j][k]], 
                                                                                                             count,
                                                                                                             G, 
                                                                                                             branch_first, 
                                                                                                             branch_position, 
                                                                                                             column-2,
                                                                                                             branch_net[j],
                                                                                                             initializer)
                            #print(branch_second, current_in_shape_base)
                            branch_first = branch_second
                            small_net.append(branch_net[j]);
                        else:
                            branch_second, count = get_layer_uid(network_stack[i][j][k], count);
                            branch_net[j] = custom_model_get_layer(network_stack[i][j][k], initializer)(branch_net[j]);
                            small_net.append(custom_model_get_layer(network_stack[i][j][k], initializer));
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
                    out = keras.layers.concatenate(branch_net)
                else:
                    out = keras.layers.add(branch_net)

                sequential_second, count = get_layer_uid(network_stack[i][-1], count);

                G.add_node(sequential_second, pos=(2, position));
                position += 1;
                for i in range(len(branch_end_points)):
                    G.add_edge(branch_end_points[i], sequential_second);
                sequential_first = sequential_second;


        else:
            sequential_second, count = get_layer_uid(network_stack[i], count);
            G.add_node(sequential_second, pos=(2, position))
            out = custom_model_get_layer(network_stack[i], initializer)(out);
            position += 1;
            G.add_edge(sequential_first, sequential_second);
            sequential_first = sequential_second;


    net = keras.models.Model(inputs=input1, outputs=out)

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
        out (placeholder): Keras placeholder temporary input for this sub-network
        initializer (keras iniatialzer): Random wieght initializer for layers in the sub-network
    
    Returns:
        neural network: The required sub-branch
        directed graph: Updated directed graph
        str: Name of the outermost layer in the sub-network
        int: Vertical position of the outer most layer in the sub-network 
        int: Horizontal position of the outer most layer in the sub-network 
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
        initializer (keras iniatialzer): Random wieght initializer for layers in the sub-network
    
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