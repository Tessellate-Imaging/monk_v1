from gluon.models.imports import *
from system.imports import *

@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=True)
def get_layer(network_layer):
    layer_name = network_layer["name"];
    layer_params = network_layer["params"];
    num_ftrs = 0;
    if(layer_name == "linear"):
        layer = nn.Dense(layer_params["out_features"], weight_initializer=init.Xavier());
        num_ftrs = layer_params["out_features"];

    elif(layer_name == "dropout"):
        layer = nn.Dropout(layer_params["p"]);
    
    elif(layer_name == "relu"):
        layer = nn.Activation('relu');

    elif(layer_name == "sigmoid"):
        layer = nn.Activation('sigmoid');

    elif(layer_name == "tanh"):
        layer = nn.Activation('tanh');

    elif(layer_name == "softplus"):
        layer = nn.Activation('softrelu');

    elif(layer_name == "leakyrelu"):
        layer = nn.LeakyReLU(alpha=layer_params["negative_slope"]);

    elif(layer_name == "prelu"):
        layer = nn.PReLU(alpha_initializer=init.Xavier());

    elif(layer_name == "elu"):
        layer = nn.ELU(alpha=layer_params["alpha"]);

    elif(layer_name == "selu"):
        layer = nn.SELU();

    elif(layer_name == "swish"):
        layer = nn.Swish(beta=layer_params["beta"]);

    return layer


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



@accepts(dict, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_sigmoid(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "sigmoid";
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


@accepts(dict, beta=[int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_swish(system_dict, beta=1.0, final_layer=False):
    tmp = {};
    tmp["name"] = "swish";
    tmp["params"] = {};
    tmp["params"]["beta"] = beta;
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