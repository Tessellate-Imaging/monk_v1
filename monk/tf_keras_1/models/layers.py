from tf_keras_1.models.imports import *
from system.imports import *
from tf_keras_1.models.initializers import get_initializer



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









@accepts(dict, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=True)
def custom_model_get_layer(network_layer, network_initializer):
    layer_name = network_layer["name"];
    layer_params = network_layer["params"];
    if(layer_name == "convolution1d"):
        return custom_model_layer_convolution1d(layer_params, network_initializer);
    elif(layer_name == "convolution2d"):
        return custom_model_layer_convolution2d(layer_params, network_initializer);
    elif(layer_name == "convolution3d"):
        return custom_model_layer_convolution3d(layer_params, network_initializer);

    elif(layer_name == "transposed_convolution2d"):
        return custom_model_layer_transposed_convolution2d(layer_params, network_initializer);
    elif(layer_name == "transposed_convolution3d"):
        return custom_model_layer_transposed_convolution3d(layer_params, network_initializer);

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

    elif(layer_name == "flatten"):
        return custom_model_layer_flatten(layer_params);
    elif(layer_name == "fully_connected"):
        return custom_model_layer_fully_connected(layer_params, network_initializer);
    elif(layer_name == "dropout"):
        return custom_model_layer_dropout(layer_params);
    elif(layer_name == "identity"):
        return custom_model_layer_identity(layer_params);

    elif(layer_name == "batch_normalization"):
        return custom_model_layer_batch_normalization(layer_params);

    elif(layer_name == "relu"):
        return custom_model_activation_relu(layer_params);
    elif(layer_name == "softmax"):
        return custom_model_activation_softmax(layer_params);
    elif(layer_name == "thresholded_relu"):
        return custom_model_activation_thresholded_relu(layer_params);
    elif(layer_name == "elu"):
        return custom_model_activation_elu(layer_params);
    elif(layer_name == "prelu"):
        return custom_model_activation_prelu(layer_params);
    elif(layer_name == "leaky_relu"):
        return custom_model_activation_leaky_relu(layer_params);
    elif(layer_name == "selu"):
        return custom_model_activation_selu(layer_params);
    elif(layer_name == "softplus"):
        return custom_model_activation_softplus(layer_params);
    elif(layer_name == "softsign"):
        return custom_model_activation_softsign(layer_params);
    elif(layer_name == "tanh"):
        return custom_model_activation_tanh(layer_params);
    elif(layer_name == "sigmoid"):
        return custom_model_activation_sigmoid(layer_params);
    elif(layer_name == "hard_sigmoid"):
        return custom_model_activation_hard_sigmoid(layer_params);






@accepts(dict, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_convolution1d(params, network_initializer):
    out_channels = params["output_channels"];
    kernel_size=params["kernel_size"];
    strides=1;
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';

    dilation_rate=params["dilation"];
    activation=None;
    use_bias = params["use_bias"];
    kernel_initializer=get_initializer(network_initializer)
    bias_initializer='zeros';
    kernel_regularizer=None;
    bias_regularizer=None;
    activity_regularizer=None;
    kernel_constraint=None;
    bias_constraint=None;


    layer = keras.layers.Conv1D(out_channels, kernel_size, strides=strides, 
                            padding=padding, data_format=data_format, 
                            dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, 
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, 
                            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, 
                            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, 
                            bias_constraint=bias_constraint)

    return layer


@accepts(dict, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_convolution2d(params, network_initializer):
    out_channels = params["output_channels"];
    kernel_size=params["kernel_size"];
    strides=1;
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';

    dilation_rate=params["dilation"];
    activation=None;
    use_bias = params["use_bias"];
    kernel_initializer=get_initializer(network_initializer);
    bias_initializer='zeros';
    kernel_regularizer=None;
    bias_regularizer=None;
    activity_regularizer=None;
    kernel_constraint=None;
    bias_constraint=None;


    layer = keras.layers.Conv2D(out_channels, kernel_size, strides=strides, 
                            padding=padding, data_format=data_format, 
                            dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, 
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, 
                            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, 
                            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, 
                            bias_constraint=bias_constraint)

    return layer



@accepts(dict, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_convolution3d(params, network_initializer):
    out_channels = params["output_channels"];
    kernel_size=params["kernel_size"];
    strides=1;
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';

    dilation_rate=params["dilation"];
    activation=None;
    use_bias = params["use_bias"];
    kernel_initializer=get_initializer(network_initializer);
    bias_initializer='zeros';
    kernel_regularizer=None;
    bias_regularizer=None;
    activity_regularizer=None;
    kernel_constraint=None;
    bias_constraint=None;


    layer = keras.layers.Conv3D(out_channels, kernel_size, strides=strides, 
                            padding=padding, data_format=data_format, 
                            dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, 
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, 
                            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, 
                            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, 
                            bias_constraint=bias_constraint)

    return layer



@accepts(dict, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_transposed_convolution2d(params, network_initializer):
    out_channels = params["output_channels"];
    kernel_size=params["kernel_size"];
    strides=1;
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';

    dilation_rate=params["dilation"];
    activation=None;
    use_bias = params["use_bias"];
    kernel_initializer=get_initializer(network_initializer);
    bias_initializer='zeros';
    kernel_regularizer=None;
    bias_regularizer=None;
    activity_regularizer=None;
    kernel_constraint=None;
    bias_constraint=None;


    layer = keras.layers.Conv2DTranspose(out_channels, kernel_size, strides=strides, 
                            padding=padding, data_format=data_format, 
                            dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, 
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, 
                            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, 
                            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, 
                            bias_constraint=bias_constraint)

    return layer


@accepts(dict, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_transposed_convolution3d(params, network_initializer):
    out_channels = params["output_channels"];
    kernel_size=params["kernel_size"];
    strides=1;
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';

    dilation_rate=params["dilation"];
    activation=None;
    use_bias = params["use_bias"];
    kernel_initializer=get_initializer(network_initializer);
    bias_initializer='zeros';
    kernel_regularizer=None;
    bias_regularizer=None;
    activity_regularizer=None;
    kernel_constraint=None;
    bias_constraint=None;


    layer = keras.layers.Conv3DTranspose(out_channels, kernel_size, strides=strides, 
                            padding=padding, data_format=data_format, 
                            dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, 
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, 
                            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, 
                            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, 
                            bias_constraint=bias_constraint)
    
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_max_pooling1d(params):
    pool_size=params["kernel_size"];
    strides=params["stride"];
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';


    layer = keras.layers.MaxPooling1D(pool_size=pool_size, strides=strides, 
                                    padding=padding, data_format=data_format)

    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_max_pooling2d(params):
    pool_size=params["kernel_size"];
    strides=params["stride"];
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';


    layer = keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, 
                                    padding=padding, data_format=data_format)

    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_max_pooling3d(params):
    pool_size=params["kernel_size"];
    strides=params["stride"];
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';


    layer = keras.layers.MaxPooling3D(pool_size=pool_size, strides=strides, 
                                    padding=padding, data_format=data_format)

    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_average_pooling1d(params):
    pool_size=params["kernel_size"];
    strides=params["stride"];
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';


    layer = keras.layers.AveragePooling1D(pool_size=pool_size, strides=strides, 
                                    padding=padding, data_format=data_format)

    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_average_pooling2d(params):
    pool_size=params["kernel_size"];
    strides=params["stride"];
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';


    layer = keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, 
                                    padding=padding, data_format=data_format)

    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_average_pooling3d(params):
    pool_size=params["kernel_size"];
    strides=params["stride"];
    if(params["padding"] == "in_eq_out"):
        padding = "same";
    elif(params["padding"] == 0):
        padding = "valid";
    else:
        padding = "causal";

    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';


    layer = keras.layers.AveragePooling3D(pool_size=pool_size, strides=strides, 
                                    padding=padding, data_format=data_format)

    return layer



@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_max_pooling1d(params):
    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';

    layer = keras.layers.GlobalMaxPooling1D(data_format=data_format);
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_max_pooling2d(params):
    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';
    
    layer = keras.layers.GlobalMaxPooling2D(data_format=data_format);
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_max_pooling3d(params):
    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';
    
    layer = keras.layers.GlobalMaxPooling3D(data_format=data_format);
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_average_pooling1d(params):
    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';

    layer = keras.layers.GlobalAveragePooling1D(data_format=data_format);
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_average_pooling2d(params):
    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';

    layer = keras.layers.GlobalAveragePooling2D(data_format=data_format);
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_global_average_pooling3d(params):
    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';

    layer = keras.layers.GlobalAveragePooling3D(data_format=data_format);
    return layer


@accepts(dict, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_flatten(params):
    if(params["layout"][-1] == "C"):
        data_format='channels_last';
    else:
        data_format='channels_first';

    layer = keras.layers.Flatten(data_format=data_format);
    return layer;



@accepts(dict, str, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_fully_connected(params, network_initializer):
    units=params["units"];
    activation=None;
    use_bias = params["use_bias"];
    kernel_initializer=get_initializer(network_initializer);
    bias_initializer='zeros';
    kernel_regularizer=None;
    bias_regularizer=None;
    activity_regularizer=None;
    kernel_constraint=None;
    bias_constraint=None;


    layer = keras.layers.Dense(units, activation=activation, use_bias=use_bias, 
                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, 
                   kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, 
                   activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, 
                   bias_constraint=bias_constraint)

    
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_dropout(params):
    rate=params["drop_probability"];
    layer = keras.layers.Dropout(rate, noise_shape=None, seed=None);

    return layer;


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_identity(params):
    layer = keras.activations.linear;
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_layer_batch_normalization(params):
    axis = -1;
    momentum=params["moving_average_momentum"];
    epsilon=params["epsilon"];
    center=params["use_trainable_parameters"];
    scale=params["use_trainable_parameters"];
    beta_initializer='zeros';
    gamma_initializer='ones';
    moving_mean_initializer='zeros';
    moving_variance_initializer='ones';
    beta_regularizer=None;
    gamma_regularizer=None;
    beta_constraint=None;
    gamma_constraint=None;



    layer = keras.layers.BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon, 
                                center=center, scale=scale, beta_initializer=beta_initializer, 
                                gamma_initializer=gamma_initializer, 
                                moving_mean_initializer=moving_mean_initializer, 
                                moving_variance_initializer=moving_variance_initializer, 
                                beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, 
                                beta_constraint=beta_constraint, gamma_constraint=gamma_constraint);


    return layer;



@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_relu(params):
    layer = keras.layers.ReLU();
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_softmax(params):
    axis = params["axis"];

    layer = keras.layers.Softmax(axis=axis);
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_thresholded_relu(params):
    threshold = params["threshold"];

    layer = keras.layers.ThresholdedReLU(theta=threshold);
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_elu(params):
    alpha = params["alpha"];
    
    layer = keras.layers.ELU(alpha=alpha);
    return layer



@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_prelu(params):
    layer = keras.layers.PReLU();
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_leaky_relu(params):
    alpha = params["alpha"];
    
    layer = keras.layers.LeakyReLU(alpha=alpha);
    return layer



@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_selu(params):
    layer = keras.layers.Activation('selu');
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_softplus(params):
    layer =keras.layers.Activation('softplus');
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_softsign(params):
    layer = keras.layers.Activation('softsign');
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_tanh(params):
    layer = keras.layers.Activation('tanh');
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_sigmoid(params):
    layer = keras.layers.Activation('sigmoid');
    return layer


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=True, trace_rv=False)
def custom_model_activation_hard_sigmoid(params):
    layer = keras.layers.Activation('hard_sigmoid');
    return layer


