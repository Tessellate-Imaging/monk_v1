from monk.gluon.models.imports import *
from monk.system.imports import *

@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=True)
def get_layer(network_layer):
    '''
    Get layer for appending it to transfer learning base model

    Args:
        network_layer (dict): Dictionary conatining all params relating to the layer

    Returns:
        neural network layer
    '''
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


@accepts(dict, probability=float, final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def layer_dropout(system_dict, probability=0.5, final_layer=False):
    '''
    Append dropout layer to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        probability (float): Droping probability of neurons in next layer
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "dropout";
    tmp["params"] = {};
    tmp["params"]["p"] = probability;
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;


@accepts(dict, num_neurons=int, final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def layer_linear(system_dict, num_neurons=512, final_layer=False):
    '''
    Append dense (fully connected) layer to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        num_neurons (int): Number of neurons in the dense layer
        final_layer (bool): If True, then number of neurons are directly set as number of classes in dataset for single label type classification

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "linear";
    tmp["params"] = {};
    tmp["params"]["out_features"] = num_neurons;
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;





@accepts(dict, alpha=[int, float], final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def activation_elu(system_dict, alpha=1.0, final_layer=False):
    '''
    Append exponential linear unit activation to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        alpha (float): Multiplicatve factor.
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "elu";
    tmp["params"] = {};
    tmp["params"]["alpha"] = alpha;
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;

@accepts(dict, negative_slope=[int, float], final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def activation_leakyrelu(system_dict, negative_slope=0.01, final_layer=False):
    '''
    Append Leaky - ReLU activation to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        negative_slope (float): Multiplicatve factor towards negative spectrum of real numbers.
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "leakyrelu";
    tmp["params"] = {};
    tmp["params"]["negative_slope"] = negative_slope; 
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;


@accepts(dict, num_parameters=int, init=[int, float], final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def activation_prelu(system_dict, num_parameters=1, init=0.25, final_layer=False):
    '''
    Append Learnable parameterized rectified linear unit activation to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        init (float): Initialization value for multiplicatve factor towards negative spectrum of real numbers.
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "prelu";
    tmp["params"] = {};
    tmp["params"]["num_parameters"] = num_parameters; 
    tmp["params"]["init"] = init; 
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;


@accepts(dict, final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def activation_relu(system_dict, final_layer=False):
    '''
    Append rectified linear unit activation to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "relu";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;


@accepts(dict, final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def activation_selu(system_dict, final_layer=False):
    '''
    Append scaled exponential linear unit activation to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "selu";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;



@accepts(dict, final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def activation_sigmoid(system_dict, final_layer=False):
    '''
    Append sigmoid activation to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "sigmoid";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;


@accepts(dict, beta=[int, float], threshold=[int, float], final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def activation_softplus(system_dict, beta=1, threshold=20, final_layer=False):
    '''
    Append softplus activation to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        beta (int): Multiplicative factor
        threshold (int): softplus (thresholded relu) limit 
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "softplus";
    tmp["params"] = {};
    tmp["params"]["beta"] = beta; 
    tmp["params"]["threshold"] = threshold; 
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;


@accepts(dict, final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def activation_softsign(system_dict, final_layer=False):
    '''
    Append softsign activation to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "softsign";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;


@accepts(dict, beta=[int, float], final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def activation_swish(system_dict, beta=1.0, final_layer=False):
    '''
    Append swish activation to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        beta (bool): Multiplicative factor
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "swish";
    tmp["params"] = {};
    tmp["params"]["beta"] = beta;
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;

    return system_dict;


@accepts(dict, final_layer=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def activation_tanh(system_dict, final_layer=False):
    '''
    Append tanh activation to base network in transfer learning

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        final_layer (bool): Indicator that this layer marks the end of network.

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["name"] = "tanh";
    tmp["params"] = {};
    system_dict["model"]["custom_network"].append(tmp);
    system_dict["model"]["final_layer"] = final_layer;
    
    return system_dict;




@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=True)
def custom_model_get_layer(network_layer):
    '''
    Get layer for custom network design

    Args:
        network_layer (dict): Network layer as dict

    Returns:
        neural network layer: Actual layer in native backend library
    '''
    layer_name = network_layer["name"];
    layer_params = network_layer["params"];
    if(layer_name == "convolution1d"):
        return custom_model_layer_convolution1d(layer_params);
    elif(layer_name == "convolution2d"):
        return custom_model_layer_convolution2d(layer_params);
    elif(layer_name == "convolution3d"):
        return custom_model_layer_convolution3d(layer_params);

    elif(layer_name == "transposed_convolution1d"):
        return custom_model_layer_transposed_convolution1d(layer_params);
    elif(layer_name == "transposed_convolution2d"):
        return custom_model_layer_transposed_convolution2d(layer_params);
    elif(layer_name == "transposed_convolution3d"):
        return custom_model_layer_transposed_convolution3d(layer_params);

    elif(layer_name == "max_pooling1d"):
        return custom_model_layer_max_pooling1d(layer_params);
    elif(layer_name == "max_pooling2d"):
        return custom_model_layer_max_pooling2d(layer_params);
    elif(layer_name == "max_pooling3d"):
        return custom_model_layer_max_pooling3d(layer_params);

    elif(layer_name == "average_pooling1d"):
        return custom_model_layer_average_pooling1d(layer_params);
    elif(layer_name == "average_pooling2d"):
        return custom_model_layer_average_pooling2d(layer_params);
    elif(layer_name == "average_pooling3d"):
        return custom_model_layer_average_pooling3d(layer_params);

    elif(layer_name == "global_max_pooling1d"):
        return custom_model_layer_global_max_pooling1d(layer_params);
    elif(layer_name == "global_max_pooling2d"):
        return custom_model_layer_global_max_pooling2d(layer_params);
    elif(layer_name == "global_max_pooling3d"):
        return custom_model_layer_global_max_pooling3d(layer_params);

    elif(layer_name == "global_average_pooling1d"):
        return custom_model_layer_global_average_pooling1d(layer_params);
    elif(layer_name == "global_average_pooling2d"):
        return custom_model_layer_global_average_pooling2d(layer_params);
    elif(layer_name == "global_average_pooling3d"):
        return custom_model_layer_global_average_pooling3d(layer_params);

    elif(layer_name == "fully_connected"):
        return custom_model_layer_fully_connected(layer_params);
    elif(layer_name == "dropout"):
        return custom_model_layer_dropout(layer_params);
    elif(layer_name == "flatten"):
        return custom_model_layer_flatten(layer_params);
    elif(layer_name == "identity"):
        return custom_model_layer_identity(layer_params);

    elif(layer_name == "batch_normalization"):
        return custom_model_layer_batch_normalization(layer_params);
    elif(layer_name == "instance_normalization"):
        return custom_model_layer_instance_normalization(layer_params);
    elif(layer_name == "layer_normalization"):
        return custom_model_layer_layer_normalization(layer_params);

    elif(layer_name == "relu"):
        return custom_model_activation_relu(layer_params);
    elif(layer_name == "sigmoid"):
        return custom_model_activation_sigmoid(layer_params);
    elif(layer_name == "tanh"):
        return custom_model_activation_tanh(layer_params);
    elif(layer_name == "softplus"):
        return custom_model_activation_softplus(layer_params);
    elif(layer_name == "softsign"):
        return custom_model_activation_softsign(layer_params);

    elif(layer_name == "elu"):
        return custom_model_activation_elu(layer_params);
    elif(layer_name == "gelu"):
        return custom_model_activation_gelu(layer_params);
    elif(layer_name == "leaky_relu"):
        return custom_model_activation_leaky_relu(layer_params);
    elif(layer_name == "prelu"):
        return custom_model_activation_prelu(layer_params);
    elif(layer_name == "selu"):
        return custom_model_activation_selu(layer_params);
    elif(layer_name == "swish"):
        return custom_model_activation_swish(layer_params);







@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_convolution1d(params):
    '''
    Append 1d-convolution to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 1d-convolution layer
    '''
    if(params["padding"] == "in_eq_out" and params["stride"]==1):
        params["padding"] = (params["dilation"]*(params["kernel_size"] - 1) - params["stride"] + 1)//2;
    elif(params["padding"] == "in_eq_out" and params["stride"]!=1):
        params["padding"] = 0;

    layer = nn.Conv1D(params["output_channels"], 
                        params["kernel_size"], 
                        strides=params["stride"], 
                        padding=params["padding"], 
                        dilation=params["dilation"], 
                        groups=params["groups"], 
                        layout=params["layout"], 
                        activation=None, 
                        use_bias=params["use_bias"], 
                        weight_initializer=None, 
                        bias_initializer='zeros', 
                        in_channels=0);
    return layer



@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_convolution2d(params):
    '''
    Append 2d-convolution to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 2d-convolution layer
    '''
    if(params["padding"] == "in_eq_out" and params["stride"]==1):
        params["padding"] = (params["dilation"]*(params["kernel_size"] - 1) - params["stride"] + 1)//2;
    elif(params["padding"] == "in_eq_out" and params["stride"]!=1):
        params["padding"] = 0;

    layer = nn.Conv2D(params["output_channels"], 
                        params["kernel_size"], 
                        strides=params["stride"], 
                        padding=params["padding"], 
                        dilation=params["dilation"], 
                        groups=params["groups"], 
                        layout=params["layout"], 
                        activation=None, 
                        use_bias=params["use_bias"], 
                        weight_initializer=None, 
                        bias_initializer='zeros', 
                        in_channels=0);
    return layer



@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_convolution3d(params):
    '''
    Append 3d-convolution to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 3d-convolution layer
    '''
    if(params["padding"] == "in_eq_out" and params["stride"]==1):
        params["padding"] = (params["dilation"]*(params["kernel_size"] - 1) - params["stride"] + 1)//2;
    elif(params["padding"] == "in_eq_out" and params["stride"]!=1):
        params["padding"] = 0;
        
    layer = nn.Conv3D(params["output_channels"], 
                        params["kernel_size"], 
                        strides=params["stride"], 
                        padding=params["padding"], 
                        dilation=params["dilation"], 
                        groups=params["groups"], 
                        layout=params["layout"], 
                        activation=None, 
                        use_bias=params["use_bias"], 
                        weight_initializer=None, 
                        bias_initializer='zeros', 
                        in_channels=0);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_transposed_convolution1d(params):
    '''
    Append 1d-transposed-convolution to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 1d-transposed-convolution layer
    '''
    if(params["padding"] == "in_eq_out" and params["stride"]==1):
        params["padding"] = (params["kernel_size"] + params["output_padding"])//2;
    elif(params["padding"] == "in_eq_out" and params["stride"]!=1):
        params["padding"] = 0;
    layer = nn.Conv1DTranspose(params["output_channels"], 
                                params["kernel_size"], 
                                strides=params["stride"], 
                                padding=params["padding"], 
                                dilation=params["dilation"], 
                                groups=params["groups"], 
                                layout=params["layout"], 
                                activation=None, 
                                use_bias=params["use_bias"], 
                                weight_initializer=None, 
                                bias_initializer='zeros', 
                                in_channels=0)
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_transposed_convolution2d(params):
    '''
    Append 2d-transposed-convolution to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 2d-transposed-convolution layer
    '''
    if(params["padding"] == "in_eq_out" and params["stride"]==1):
        params["padding"] = (params["kernel_size"] + params["output_padding"])//2;
    elif(params["padding"] == "in_eq_out" and params["stride"]!=1):
        params["padding"] = 0;
    layer = nn.Conv2DTranspose(params["output_channels"], 
                                params["kernel_size"], 
                                strides=params["stride"], 
                                padding=params["padding"], 
                                dilation=params["dilation"], 
                                groups=params["groups"], 
                                layout=params["layout"], 
                                activation=None, 
                                use_bias=params["use_bias"], 
                                weight_initializer=None, 
                                bias_initializer='zeros', 
                                in_channels=0)
    return layer



@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_transposed_convolution3d(params):
    '''
    Append 3d-transposed-convolution to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 3d-transposed-convolution layer
    '''
    if(params["padding"] == "in_eq_out" and params["stride"]==1):
        params["padding"] = (params["kernel_size"] + params["output_padding"])//2;
    elif(params["padding"] == "in_eq_out" and params["stride"]!=1):
        params["padding"] = 0;
    layer = nn.Conv3DTranspose(params["output_channels"], 
                                params["kernel_size"], 
                                strides=params["stride"], 
                                padding=params["padding"], 
                                dilation=params["dilation"], 
                                groups=params["groups"], 
                                layout=params["layout"], 
                                activation=None, 
                                use_bias=params["use_bias"], 
                                weight_initializer=None, 
                                bias_initializer='zeros', 
                                in_channels=0)
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_max_pooling1d(params):
    '''
    Append 1d-max-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 1d-max-pooling layer
    '''
    layer = nn.MaxPool1D(pool_size=params["kernel_size"], 
                        strides=params["stride"], 
                        padding=params["padding"],
                        ceil_mode=params["ceil_mode"],
                        layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_max_pooling2d(params):
    '''
    Append 2d-max-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 2d-max-pooling layer
    '''
    layer = nn.MaxPool2D(pool_size=params["kernel_size"], 
                        strides=params["stride"], 
                        padding=params["padding"],
                        ceil_mode=params["ceil_mode"],
                        layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_max_pooling3d(params):
    '''
    Append 3d-max-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 3d-max-pooling layer
    '''
    layer = nn.MaxPool3D(pool_size=params["kernel_size"], 
                        strides=params["stride"], 
                        padding=params["padding"],
                        ceil_mode=params["ceil_mode"],
                        layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_average_pooling1d(params):
    '''
    Append 1d-average-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 1d-average-pooling layer
    '''
    layer = nn.AvgPool1D(pool_size=params["kernel_size"], 
                        strides=params["stride"], 
                        padding=params["padding"],
                        ceil_mode=params["ceil_mode"],
                        count_include_pad=params["include_padding_in_calculation"],
                        layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_average_pooling2d(params):
    '''
    Append 2d-average-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 2d-average-pooling layer
    '''

    layer = nn.AvgPool2D(pool_size=params["kernel_size"], 
                        strides=params["stride"], 
                        padding=params["padding"],
                        ceil_mode=params["ceil_mode"],
                        count_include_pad=params["include_padding_in_calculation"],
                        layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_average_pooling3d(params):
    '''
    Append 3d-average-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 3d-average-pooling layer
    '''

    layer = nn.AvgPool3D(pool_size=params["kernel_size"], 
                        strides=params["stride"], 
                        padding=params["padding"],
                        ceil_mode=params["ceil_mode"],
                        count_include_pad=params["include_padding_in_calculation"],
                        layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_max_pooling1d(params):
    '''
    Append 1d-global-max-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 1d-global-max-pooling layer
    '''
    layer = nn.GlobalMaxPool1D(layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_max_pooling2d(params):
    '''
    Append 2d-global-max-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 2d-global-max-pooling layer
    '''
    layer = nn.GlobalMaxPool2D(layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_max_pooling3d(params):
    '''
    Append 3d-global-max-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 3d-global-max-pooling layer
    '''
    layer = nn.GlobalMaxPool3D(layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_average_pooling1d(params):
    '''
    Append 1d-global-average-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 1d-global-average-pooling layer
    '''
    layer = nn.GlobalAvgPool1D(layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_average_pooling2d(params):
    '''
    Append 2d-global-average-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 2d-global-average-pooling layer
    '''
    layer = nn.GlobalAvgPool2D(layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_average_pooling3d(params):
    '''
    Append 3d-global-average-pooling to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: 3d-global-average-pooling layer
    '''
    layer = nn.GlobalAvgPool3D(layout=params["layout"]);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_fully_connected(params):
    '''
    Append fc (dense) to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: fc (dense) layer
    '''
    layer = nn.Dense(params["units"], 
                    activation=None, 
                    use_bias=params["use_bias"], 
                    flatten=params["flatten"], 
                    dtype='float32', 
                    weight_initializer=None, 
                    bias_initializer='zeros', 
                    in_units=0);
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_dropout(params):
    '''
    Append dropout to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: dropout layer
    '''
    layer = nn.Dropout(params["drop_probability"],
                        axes=params["axes"]);
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_flatten(params):
    '''
    Append flatten to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: flatten layer
    '''
    layer = nn.Flatten();
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_identity(params):
    '''
    Append idenity to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: idenity layer
    '''
    layer = contrib_nn.Identity();
    return layer


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_batch_normalization(params):
    '''
    Append batchnorm to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural netwrk layer: batchnorm layer
    '''
    layer = nn.BatchNorm(axis=1, 
                        momentum=params["moving_average_momentum"], 
                        epsilon=params["epsilon"], 
                        center=params["use_trainable_parameters"],
                        scale=params["use_trainable_parameters"], 
                        use_global_stats=params["activate_scale_shift_operation"]);
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_instance_normalization(params):
    '''
    Append instancenorm to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: instancenorm layer
    '''
    layer = nn.InstanceNorm(axis=1,
                        epsilon=params["epsilon"], 
                        center=params["use_trainable_parameters"],
                        scale=params["use_trainable_parameters"]);
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_layer_normalization(params):
    '''
    Append layernorm to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network layer: layernorm layer
    '''
    layer = nn.LayerNorm(axis=1, 
                        epsilon=params["epsilon"], 
                        center=params["use_trainable_parameters"],
                        scale=params["use_trainable_parameters"]);
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_relu(params):
    '''
    Append relu to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: relu activation
    '''
    layer = nn.Activation("relu");
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_sigmoid(params):
    '''
    Append sigmoid to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: sigmoid activation
    '''
    layer = nn.Activation("sigmoid");
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_tanh(params):
    '''
    Append tanh to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: tanh activation
    '''
    layer = nn.Activation("tanh");
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_softplus(params):
    '''
    Append softplus to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: softplus activation
    '''
    layer = nn.Activation("softrelu");
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_softsign(params):
    '''
    Append softsign to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: softsign activation
    '''
    layer = nn.Activation("softsign");
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_elu(params):
    '''
    Append elu to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: elu activation
    '''
    layer = nn.ELU(alpha=params["alpha"]);
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_gelu(params):
    '''
    Append gelu to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: gelu activation
    '''
    layer = nn.GELU();
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_prelu(params):
    '''
    Append prelu to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: prelu activation
    '''
    layer = nn.PReLU();
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_leaky_relu(params):
    '''
    Append leaky-relu to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: leaky-relu activation
    '''
    layer = nn.LeakyReLU(alpha=params["alpha"]);
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_selu(params):
    '''
    Append selu to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: selu activation
    '''
    layer = nn.SELU();
    return layer;


@accepts(dict, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_swish(params):
    '''
    Append swish to custom network

    Args:
        params (dict): All layer parameters

    Returns:
        neural network activation: swish activation
    '''
    layer = nn.Swish(beta=params["beta"]);
    return layer;



class addBlock(nn.HybridBlock):
    '''
    Class for creating a custom block (layer) representing elementwise addition

    Args:
        branches (list): List of layers to merge with elementwise addition
    '''
    def __init__(self, branches, **kwargs):
        super(addBlock, self).__init__(**kwargs)
        self.child_names = [];     
        
        with self.name_scope():
            for i in range(len(branches)):
                vars(self)["body" + str(i)] = nn.HybridSequential(prefix='');
                for j in range(len(branches[i])):
                    vars(self)["body" + str(i)].add(branches[i][j]);
                self.child_names.append("body" + str(i));
        
        for i, child in enumerate(self.child_names):
            setattr(self, 'body{0}'.format(i), vars(self)["body" + str(i)])
        
            
    def hybrid_forward(self, F, x):
        for i in range(len(self.child_names)):
            if(i==0):
                y = vars(self)["body" + str(i)](x);
            else:
                y = y + vars(self)["body" + str(i)](x);
            
        return y













    


