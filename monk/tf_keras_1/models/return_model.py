from tf_keras_1.models.imports import *
from system.imports import *
from tf_keras_1.models.models import *
from tf_keras_1.models.common import create_final_layer
from tf_keras_1.models.common import get_layer_uid
from tf_keras_1.models.layers import custom_model_get_layer



@accepts(dict, path=[str, bool], final=bool, resume=bool, external_path=[bool, str, list], post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def load_model(system_dict, path=False, final=False, resume=False, external_path=False):
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






@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def setup_model(system_dict):
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
            x = krl.Dense(num_classes)(x);
            preds = krl.Softmax()(x);
            finetune_net = keras.models.Model(inputs=finetune_net.input, outputs=preds);

        system_dict["local"]["model"] = finetune_net;

        return system_dict;


    else:
        count = [];
        for i in range(len(names)):
            count.append(1);

        if(len(system_dict["dataset"]["params"]["data_shape"]) == 2):
            c, w = system_dict["dataset"]["params"]["data_shape"];
            input1 = krl.Input(shape=(w, c));
        elif(len(system_dict["dataset"]["params"]["data_shape"]) == 3):
            c, h, w = system_dict["dataset"]["params"]["data_shape"];
            input1 = krl.Input(shape=(h, w, c));
        elif(len(system_dict["dataset"]["params"]["data_shape"]) == 4):
            c, d, h, w = system_dict["dataset"]["params"]["data_shape"];
            input1 = krl.Input(shape=(d, h, w, c));


        

        network_stack = system_dict["custom_model"]["network_stack"];
        G=nx.DiGraph()
        G.add_node("Net", pos=(1,1))
        sequential_first = "data";
        sequential_second, count = get_layer_uid(network_stack[0], count)

        count = [];
        for i in range(len(names)):
            count.append(1);

        position = 1;
        G.add_node(sequential_first, pos=(2,1))
        position += 1;

        max_width = 1;


        for i in range(len(network_stack)):
            if(type(network_stack[i]) == list):
                branch_end_points = [];
                branch_lengths = [];
                branches = [];
                branch_net = [];
                out_base = out;

                if(max_width < len(network_stack[i])-2):
                    max_width = len(network_stack[i])-2
                for j in range(len(network_stack[i])-1):
                    small_net = [];
                    branch_net.append(out_base)
                    branch_first = sequential_first
                    branch_position = position
                    column = j+2;
                    for k in range(len(network_stack[i][j])):
                        branch_second, count = get_layer_uid(network_stack[i][j][k], count);
                        small_net.append(custom_model_get_layer(network_stack[i][j][k], system_dict["custom_model"]["network_initializer"]));
                        
                        branch_net[j] = custom_model_get_layer(network_stack[i][j][k], system_dict["custom_model"]["network_initializer"])(branch_net[j]);
                        
                        G.add_node(branch_second, pos=(column, branch_position));
                        branch_position += 1;
                        G.add_edge(branch_first, branch_second);
                        branch_first = branch_second;

                        if(k == len(network_stack[i][j])-1):
                            branch_end_points.append(branch_second);
                            branch_lengths.append(len(network_stack[i][j]));
                    branches.append(small_net);

                position += max(branch_lengths);
                position += 1;

                sequential_second, count = get_layer_uid(network_stack[i][-1], count)
                if(network_stack[i][-1]["name"] == "concatenate"):
                    out = keras.layers.concatenate(branch_net)

                else:
                    out = keras.layers.add(branch_net)

                G.add_node(sequential_second, pos=(2, position));
                position += 1;
                for i in range(len(branch_end_points)):
                    G.add_edge(branch_end_points[i], sequential_second);
                sequential_first = sequential_second;

            else:
                sequential_second, count = get_layer_uid(network_stack[i], count)
                
                # Get layer here
                if(i == 0):
                    out = custom_model_get_layer(network_stack[i], system_dict["custom_model"]["network_initializer"])(input1);
                else:
                    out = custom_model_get_layer(network_stack[i], system_dict["custom_model"]["network_initializer"])(out);

                G.add_node(sequential_second, pos=(2, position))
                position += 1;
                G.add_edge(sequential_first, sequential_second);
                sequential_first = sequential_second;


        net = keras.models.Model(inputs=input1, outputs=out)

        if(max_width == 1):
            G.add_node("monk", pos=(3, position));
        else:
            G.add_node("monk", pos=(max_width + 3, position))
        pos=nx.get_node_attributes(G,'pos')

        plt.figure(3, figsize=(8, 12 + position//6)) 
        nx.draw_networkx(G, pos, with_label=True, font_size=16, node_color="yellow", node_size=100)
        plt.savefig("graph.png");

        system_dict["local"]["model"] = net;

        return system_dict;



@accepts(list, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def debug_custom_model(network_stack):
    count = [];
    for i in range(len(names)):
        count.append(1);

    G=nx.DiGraph()
    G.add_node("Net", pos=(1,1))
    sequential_first = "data";
    sequential_second, count = get_layer_uid(network_stack[0], count)

    count = [];
    for i in range(len(names)):
        count.append(1);

    position = 1;
    G.add_node(sequential_first, pos=(2,1))
    position += 1;


    max_width = 1;
    for i in range(len(network_stack)):
        if(type(network_stack[i]) == list):
            branch_end_points = [];
            branch_lengths = [];
            branches = [];
            branch_net = [];


            if(max_width < len(network_stack[i])-2):
                max_width = len(network_stack[i])-2
            for j in range(len(network_stack[i])-1):
                branch_first = sequential_first
                branch_position = position
                column = j+2;
                for k in range(len(network_stack[i][j])):
                    branch_second, count = get_layer_uid(network_stack[i][j][k], count);
                    G.add_node(branch_second, pos=(column, branch_position));
                    branch_position += 1;
                    G.add_edge(branch_first, branch_second);
                    branch_first = branch_second;

                    if(k == len(network_stack[i][j])-1):
                        branch_end_points.append(branch_second);
                        branch_lengths.append(len(network_stack[i][j]));

            position += max(branch_lengths);
            position += 1;

            sequential_second, count = get_layer_uid(network_stack[i][-1], count)

            G.add_node(sequential_second, pos=(2, position));
            position += 1;
            for i in range(len(branch_end_points)):
                G.add_edge(branch_end_points[i], sequential_second);
            sequential_first = sequential_second;


        else:
            sequential_second, count = get_layer_uid(network_stack[i], count)
            G.add_node(sequential_second, pos=(2, position))
            position += 1;
            G.add_edge(sequential_first, sequential_second);
            sequential_first = sequential_second;


    if(max_width == 1):
        G.add_node("monk", pos=(3, position));
    else:
        G.add_node("monk", pos=(max_width + 3, position))
    pos=nx.get_node_attributes(G,'pos')

    plt.figure(3, figsize=(8, 12 + position//6)) 
    nx.draw_networkx(G, pos, with_label=True, font_size=16, node_color="yellow", node_size=100)
    plt.savefig("graph.png");