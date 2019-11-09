from pytorch.models.imports import *
from system.imports import *


@accepts(dict, int, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=True)
def get_layer(network_layer, num_ftrs):
    layer_name = network_layer["name"];
    layer_params = network_layer["params"];
    
    if(layer_name == "linear"):
        layer = nn.Linear(num_ftrs, layer_params["out_features"])
        num_ftrs = layer_params["out_features"];
    elif(layer_name == "dropout"):
        layer = nn.Dropout(p=layer_params["p"]);
    elif(layer_name == "elu"):
        layer = nn.ELU(alpha=layer_params["alpha"]);
    elif(layer_name == "hardshrink"):
        layer = nn.Hardshrink(lambd=layer_params["lambd"]);
    elif(layer_name == "hardtanh"):
        layer = nn.Hardtanh(min_val=layer_params["min_val"], max_val=layer_params["max_val"]);
    elif(layer_name == "leakyrelu"):
        layer = nn.LeakyReLU(negative_slope=layer_params["negative_slope"]);
    elif(layer_name == "logsigmoid"):
        layer = nn.LogSigmoid();
    elif(layer_name == "prelu"):
        layer = nn.PReLU(num_parameters=layer_params["num_parameters"], init=layer_params["init"]);
    elif(layer_name == "relu"):
        layer = nn.ReLU();
    elif(layer_name == "relu6"):
        layer = nn.ReLU6();
    elif(layer_name == "rrelu"):
        layer = nn.RReLU(lower=layer_params["lower"], upper=layer_params["upper"]);
    elif(layer_name == "selu"):
        layer = nn.SELU();
    elif(layer_name == "celu"):
        layer = nn.CELU(alpha=layer_params["alpha"]);
    elif(layer_name == "sigmoid"):
        layer = nn.Sigmoid();
    elif(layer_name == "softplus"):
        layer = nn.Softplus(beta=layer_params["beta"], threshold=layer_params["threshold"]);
    elif(layer_name == "softshrink"):
        layer = nn.Softshrink(lambd=layer_params["lambd"]);
    elif(layer_name == "softsign"):
        layer = nn.Softsign();
    elif(layer_name == "tanh"):
        layer = nn.Tanh();
    elif(layer_name == "tanhshrink"):
        layer = nn.Tanhshrink();
    elif(layer_name == "threshold"):
        layer = nn.Threshold(threshold=layer_params["threshold"], value=layer_params["value"]);
    elif(layer_name == "softmin"):
        layer = nn.Softmin();
    elif(layer_name == "softmax"):
        layer = nn.Softmax();
    elif(layer_name == "logsoftmax"):
        layer = nn.LogSoftmax();

    return layer, num_ftrs;






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



@accepts(dict, lambd=[int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_hardshrink(system_dict, lambd=0.5, final_layer=False):
    tmp = {};
    tmp["name"] = "hardshrink";
    tmp["params"] = {};
    tmp["params"]["lambd"] = lambd;
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;


@accepts(dict, min_val=[int, float], max_val=[int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_hardtanh(system_dict, min_val=-1.0, max_val=1.0, final_layer=False):
    tmp = {};
    tmp["name"] = "hardtanh";
    tmp["params"] = {};
    tmp["params"]["min_val"] = min_val;
    tmp["params"]["max_val"] = max_val;
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



@accepts(dict, final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_logsigmoid(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "logsigmoid";
    tmp["params"] = {};
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
def activation_relu6(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "relu6";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;



@accepts(dict, lower=[int, float], upper=[int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_rrelu(system_dict, lower=0.125, upper=0.333, final_layer=False):
    tmp = {};
    tmp["name"] = "rrelu";
    tmp["params"] = {};
    tmp["params"]["lower"] = lower; 
    tmp["params"]["upper"] = upper; 
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


@accepts(dict, alpha=[int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_celu(system_dict, alpha=1.0, final_layer=False):
    tmp = {};
    tmp["name"] = "celu";
    tmp["params"] = {};
    tmp["params"]["alpha"] = alpha;
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




@accepts(dict, lambd=[int, float], final_layer=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def activation_softshrink(system_dict, lambd=0.5, final_layer=False):
    tmp = {};
    tmp["name"] = "softshrink";
    tmp["params"] = {};
    tmp["params"]["lambd"] = lambd;
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
def activation_tanhshrink(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "tanhshrink";
    tmp["params"] = {};
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
def activation_softmin(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "softmin";
    tmp["params"] = {};
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
def activation_logsoftmax(system_dict, final_layer=False):
    tmp = {};
    tmp["name"] = "logsoftmax";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;