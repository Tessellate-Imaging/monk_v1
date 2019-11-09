from tf_keras.models.imports import *
from system.imports import *




@accepts(dict, "self", post_trace=True)
@TraceFunction(trace_args=True, trace_rv=True)
def get_layer(network_layer, inp):
    layer_name = network_layer["name"];
    layer_params = network_layer["params"];


    if(layer_name == "linear"):
        out = krl.Dense(layer_params["out_features"])(inp)
    
    elif(layer_name == "dropout"):
        out = krl.Dropout(rate=1-layer_params["p"])(inp)

    elif(layer_name == "globalaveragepooling"):
        out = krl.GlobalAveragePooling2D()(inp);

    elif(layer_name == "flatten"):
        out = krl.Flatten()(inp);
    
    elif(layer_name == "elu"):
        out = krl.ELU(alpha=layer_params["alpha"])(inp);
    
    elif(layer_name == "leakyrelu"):
        out = krl.LeakyReLU(alpha=layer_params["alpha"])(inp);
    
    elif(layer_name == "prelu"):
        init = kri.Constant(value=layer_params["init"])
        out = krl.PReLU(alpha_initializer=init)(inp);
    
    elif(layer_name == "relu"):
        out = krl.ReLU()(inp);

    elif(layer_name == "selu"):
        out = krl.selu(inp);

    elif(layer_name == "sigmoid"):
        out = krl.sigmoid(inp);

    elif(layer_name == "softplus"):
        out = krl.softplus(inp);

    elif(layer_name == "softsign"):
        out = krl.softsign(inp);

    elif(layer_name == "tanh"):
        out = krl.tanh(inp);

    elif(layer_name == "threshold"):
        out = krl.ThresholdedReLU(theta=layer_params["threshold"])(inp);

    elif(layer_name == "softmax"):
        out = krl.Softmax()(inp);


    return out;



@accepts(dict, num_neurons=int, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def layer_linear(system_dict, num_neurons=512, final_layer=False):
    tmp = {};
    tmp["name"] = "linear";
    tmp["params"] = {};
    tmp["params"]["out_features"] = num_neurons;
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;



@accepts(dict, probability=float, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def layer_dropout(system_dict, probability=0.5, final_layer=False):
    tmp = {};
    tmp["name"] = "dropout";
    tmp["params"] = {};
    tmp["params"]["p"] = probability;
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;



@accepts(dict, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def layer_globalaveragepooling(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "globalaveragepooling";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;



@accepts(dict, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def layer_flatten(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "flatten";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;



@accepts(dict, negative_slope=[int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_leakyrelu(system_dict, negative_slope=0.01, final_layer=False):
    tmp = {};
    tmp["name"] = "leakyrelu";
    tmp["params"] = {};
    tmp["params"]["negative_slope"] = negative_slope; 
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;




@accepts(dict, num_parameters=int, init=[int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_prelu(system_dict, num_parameters=1, init=0.25, final_layer=False):
    tmp = {};
    tmp["name"] = "prelu";
    tmp["params"] = {};
    tmp["params"]["num_parameters"] = num_parameters; 
    tmp["params"]["init"] = init; 
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;




@accepts(dict, alpha=[int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_elu(system_dict, alpha=1.0, final_layer=False):
    tmp = {};
    tmp["name"] = "elu";
    tmp["params"] = {};
    tmp["params"]["alpha"] = alpha;
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;




@accepts(dict, [int, float], [int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_threshold(system_dict, threshold, value, final_layer=False):
    tmp = {};
    tmp["name"] = "threshold";
    tmp["params"] = {};
    tmp["params"]["value"] = value; 
    tmp["params"]["threshold"] = threshold;
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;





@accepts(dict, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_softmax(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "softmax";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;




@accepts(dict, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_relu(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "relu";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;



@accepts(dict, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_selu(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "selu";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;




@accepts(dict, beta=[int, float], threshold=[int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_softplus(system_dict, beta=1, threshold=20, final_layer=False):
    tmp = {};
    tmp["name"] = "softplus";
    tmp["params"] = {};
    tmp["params"]["beta"] = beta; 
    tmp["params"]["threshold"] = threshold; 
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;




@accepts(dict, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_softsign(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "softsign";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;




@accepts(dict, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_tanh(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "tanh";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;




@accepts(dict, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_sigmoid(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "sigmoid";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;
