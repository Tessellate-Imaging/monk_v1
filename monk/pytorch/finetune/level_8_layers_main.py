from pytorch.finetune.imports import *
from system.imports import *

from pytorch.finetune.level_7_aux_main import prototype_aux


class prototype_layers(prototype_aux):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    #####################################################################################################################################
    @warning_checks(None, num_neurons=["lt", 10000], final_layer=None, post_trace=True)
    @error_checks(None, num_neurons=["gt", 0], final_layer=None, post_trace=True)
    @accepts("self", num_neurons=[int, bool], final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_linear(self, num_neurons=False, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            if(not num_neurons):
                num_neurons = self.system_dict["dataset"]["params"]["num_classes"];
            self.system_dict = layer_linear(self.system_dict, num_neurons=num_neurons, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, probability=["lt", 0.7], final_layer=None, post_trace=True)
    @error_checks(None, probability=["gt", 0, "lt", 1], final_layer=None, post_trace=True)
    @accepts("self", probability=float, final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_dropout(self, probability=0.5, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = layer_dropout(self.system_dict, probability=probability, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", alpha=[float, int], final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_elu(self, alpha=1.0, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_elu(self.system_dict, alpha=alpha, final_layer=final_layer); 
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", alpha=[float, int], final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_hardshrink(self, lambd=0.5, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_hardshrink(self.system_dict, lambd=lambd, final_layer=final_layer);
    #####################################################################################################################################





    #####################################################################################################################################
    @accepts("self", min_val=float, max_val=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_hardtanh(self, min_val=-1.0, max_val=1.0, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_hardtanh(self.system_dict, min_val=min_val, max_val=max_val, final_layer=final_layer);
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, negative_slope=["lt", 0.2], final_layer=None, post_trace=True)
    @error_checks(None, negative_slope=["gt", 0], final_layer=None, post_trace=True)
    @accepts("self", negative_slope=[float, int], final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_leakyrelu(self, negative_slope=0.01, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_leakyrelu(self.system_dict, negative_slope=negative_slope, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_logsigmoid(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_logsigmoid(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, num_parameters=["lt", 1], final_layer=None, post_trace=True)
    @error_checks(None, num_parameters=["gt", 0], init=["gt", 0], final_layer=None, post_trace=True)
    @accepts("self", num_parameters=int, init=[int, float], final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_prelu(self, num_parameters=1, init=0.25, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_prelu(self.system_dict, num_parameters=num_parameters, init=init, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_relu(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_relu(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_relu6(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_relu6(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", lower=[int, float], upper=[int, float], final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_rrelu(self, lower=0.125, upper=0.333, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_rrelu(self.system_dict, lower=lower, upper=upper, final_layer=final_layer)
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_selu(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_selu(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", alpha=[float, int], final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_celu(self, alpha=1.0, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_celu(self.system_dict, alpha=alpha, final_layer=final_layer); 
    #####################################################################################################################################





    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_sigmoid(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_sigmoid(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", beta=[int, float], threshold=[int, float], final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_softplus(self, beta=1, threshold=20, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_softplus(self.system_dict, beta=beta, threshold=threshold, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", alpha=[float, int], final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_softshrink(self, lambd=0.5, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_softshrink(self.system_dict, lambd=lambd, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_softsign(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_softsign(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_tanh(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_tanh(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################



    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_tanhshrink(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_tanhshrink(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################



    #####################################################################################################################################
    @accepts("self", [int, float], [int, float], final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_threshold(self, threshold, value, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_threshold(self.system_dict, threshold, value, final_layer=final_layer);
    #####################################################################################################################################






    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_softmin(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_softmin(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_softmax(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_softmax(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def append_logsoftmax(self, final_layer=False):
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_logsoftmax(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, output_channels=["lt", 2048], kernel_size=["lt", 16], stride=["lt", 16], padding=None,
        groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCW"], uid=None, post_trace=True)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        groups=int, dilation=int, use_bias=bool, layout=str, uid=[type(None), str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def convolution1d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        groups=1, dilation=1, use_bias=True, layout='NCW', uid=None):
        tmp={};
        tmp["uid"] = uid;
        tmp["name"] = "convolution1d";
        tmp["params"] = {};
        tmp["params"]["output_channels"] = output_channels;
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["groups"] = groups;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["use_bias"] = use_bias;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, output_channels=["lt", 2048], kernel_size=["lt", 16], stride=["lt", 16], padding=None,
        groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        groups=int, dilation=int, use_bias=bool, layout=str, uid=[type(None), str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def convolution2d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        groups=1, dilation=1, use_bias=True, layout='NCHW', uid=None):
        tmp={};
        tmp["uid"] = uid;
        tmp["name"] = "convolution2d";
        tmp["params"] = {};
        tmp["params"]["output_channels"] = output_channels;
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["groups"] = groups;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["use_bias"] = use_bias;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, output_channels=["lt", 2048], kernel_size=["lt", 16], stride=["lt", 16], padding=None,
        groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        groups=int, dilation=int, use_bias=bool, layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def convolution(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        groups=1, dilation=1, use_bias=True, layout='NCHW', uid=None):
        tmp={};
        tmp["uid"] = uid;
        tmp["name"] = "convolution2d";
        tmp["params"] = {};
        tmp["params"]["output_channels"] = output_channels;
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["groups"] = groups;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["use_bias"] = use_bias;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, output_channels=["lt", 2048], kernel_size=["lt", 16], stride=["lt", 16], padding=None,
        groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCDHW"], uid=None, post_trace=True)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        groups=int, dilation=int, use_bias=bool, layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def convolution3d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        groups=1, dilation=1, use_bias=True, layout='NCDHW', uid=None):
        tmp={};
        tmp["uid"] = uid;
        tmp["name"] = "convolution3d";
        tmp["params"] = {};
        tmp["params"]["output_channels"] = output_channels;
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["groups"] = groups;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["use_bias"] = use_bias;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, output_channels=["lt", 2048], kernel_size=["lt", 16], stride=["lt", 16], padding=None,
        output_padding=None, groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        output_padding=["gte, 0"], groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCW"], uid=None, post_trace=True)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        output_padding=int, groups=int, dilation=int, use_bias=bool, layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def transposed_convolution1d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        output_padding=0, groups=1, dilation=1, use_bias=True, layout='NCW', uid=None):
        tmp={};
        tmp["uid"] = uid;
        tmp["name"] = "transposed_convolution1d";
        tmp["params"] = {};
        tmp["params"]["output_channels"] = output_channels;
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["output_padding"] = output_padding;
        tmp["params"]["groups"] = groups;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["use_bias"] = use_bias;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, output_channels=["lt", 2048], kernel_size=["lt", 16], stride=["lt", 16], padding=None,
        output_padding=None, groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        output_padding=["gte, 0"], groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        output_padding=int, groups=int, dilation=int, use_bias=bool, layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def transposed_convolution(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        output_padding=0, groups=1, dilation=1, use_bias=True, layout='NCHW', uid=None):
        tmp={};
        tmp["uid"] = uid;
        tmp["name"] = "transposed_convolution2d";
        tmp["params"] = {};
        tmp["params"]["output_channels"] = output_channels;
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["output_padding"] = output_padding;
        tmp["params"]["groups"] = groups;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["use_bias"] = use_bias;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, output_channels=["lt", 2048], kernel_size=["lt", 16], stride=["lt", 16], padding=None,
        output_padding=None, groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        output_padding=["gte, 0"], groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        output_padding=int, groups=int, dilation=int, use_bias=bool, layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def transposed_convolution2d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        output_padding=0, groups=1, dilation=1, use_bias=True, layout='NCHW', uid=None):
        tmp={};
        tmp["uid"] = uid;
        tmp["name"] = "transposed_convolution2d";
        tmp["params"] = {};
        tmp["params"]["output_channels"] = output_channels;
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["output_padding"] = output_padding;
        tmp["params"]["groups"] = groups;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["use_bias"] = use_bias;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, output_channels=["lt", 2048], kernel_size=["lt", 16], stride=["lt", 16], padding=None,
        output_padding=None, groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        output_padding=["gte, 0"], groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCDHW"], uid=None, post_trace=True)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        output_padding=int, groups=int, dilation=int, use_bias=bool, layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def transposed_convolution3d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        output_padding=0, groups=1, dilation=1, use_bias=True, layout='NCDHW', uid=None):
        tmp={};
        tmp["uid"] = uid;
        tmp["name"] = "transposed_convolution3d";
        tmp["params"] = {};
        tmp["params"]["output_channels"] = output_channels;
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["output_padding"] = output_padding;
        tmp["params"]["groups"] = groups;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["use_bias"] = use_bias;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, kernel_size=["lt", 16], stride=["lt", 16], padding=None, dilation=None, 
        return_indices=None, ceil_mode=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, layout=["eq", "NCW"], uid=None, post_trace=True)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def max_pooling1d(self, kernel_size=2, stride=None, padding=0, dilation=1, 
        return_indices=False, ceil_mode=False, layout='NCW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "max_pooling1d";
        tmp["params"] = {};
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["return_indices"] = return_indices;
        tmp["params"]["ceil_mode"] = ceil_mode;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################





    #####################################################################################################################################
    @warning_checks(None, kernel_size=["lt", 16], stride=["lt", 16], padding=None, dilation=None, 
        return_indices=None, ceil_mode=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int, 
        return_indices=bool, ceil_mode=bool, layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def max_pooling2d(self, kernel_size=2, stride=None, padding=0, dilation=1, 
        return_indices=False, ceil_mode=False, layout='NCHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "max_pooling2d";
        tmp["params"] = {};
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["return_indices"] = return_indices;
        tmp["params"]["ceil_mode"] = ceil_mode;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, kernel_size=["lt", 16], stride=["lt", 16], padding=None, dilation=None, 
        return_indices=None, ceil_mode=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def max_pooling(self, kernel_size=2, stride=None, padding=0, dilation=1, 
        return_indices=False, ceil_mode=False, layout='NCHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "max_pooling2d";
        tmp["params"] = {};
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["return_indices"] = return_indices;
        tmp["params"]["ceil_mode"] = ceil_mode;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, kernel_size=["lt", 16], stride=["lt", 16], padding=None, dilation=None, 
        return_indices=None, ceil_mode=None, layout=None, uid=None, post_trace=True)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, layout=["eq", "NCDHW"], uid=None, post_trace=True)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def max_pooling3d(self, kernel_size=2, stride=None, padding=0, dilation=1, 
        return_indices=False, ceil_mode=False, layout='NCDHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "max_pooling3d";
        tmp["params"] = {};
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["return_indices"] = return_indices;
        tmp["params"]["ceil_mode"] = ceil_mode;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, kernel_size=["lt", 16], stride=["lt", 16], padding=None, dilation=None, 
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None, 
        layout=None, uid=None, post_trace=True)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None,
        layout=["eq", "NCW"], uid=None, post_trace=True)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, include_padding_in_calculation=bool,
        layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def average_pooling1d(self, kernel_size=2, stride=None, padding=0, dilation=1, 
        return_indices=False, ceil_mode=False, include_padding_in_calculation=True, 
        layout='NCW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "average_pooling1d";
        tmp["params"] = {};
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["return_indices"] = return_indices;
        tmp["params"]["ceil_mode"] = ceil_mode;
        tmp["params"]["include_padding_in_calculation"] = include_padding_in_calculation;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, kernel_size=["lt", 16], stride=["lt", 16], padding=None, dilation=None, 
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None, 
        layout=None, uid=None, post_trace=True)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None,
        layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, include_padding_in_calculation=bool,
        layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def average_pooling2d(self, kernel_size=2, stride=None, padding=0, dilation=1, 
        return_indices=False, ceil_mode=False, include_padding_in_calculation=True, 
        layout='NCHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "average_pooling2d";
        tmp["params"] = {};
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["return_indices"] = return_indices;
        tmp["params"]["ceil_mode"] = ceil_mode;
        tmp["params"]["include_padding_in_calculation"] = include_padding_in_calculation;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, kernel_size=["lt", 16], stride=["lt", 16], padding=None, dilation=None, 
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None, 
        layout=None, uid=None, post_trace=True)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None,
        layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, include_padding_in_calculation=bool,
        layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def average_pooling(self, kernel_size=2, stride=None, padding=0, dilation=1, 
        return_indices=False, ceil_mode=False, include_padding_in_calculation=True, 
        layout='NCHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "average_pooling2d";
        tmp["params"] = {};
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["return_indices"] = return_indices;
        tmp["params"]["ceil_mode"] = ceil_mode;
        tmp["params"]["include_padding_in_calculation"] = include_padding_in_calculation;
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, kernel_size=["lt", 16], stride=["lt", 16], padding=None, dilation=None, 
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None, 
        layout=None, uid=None, post_trace=True)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None,
        layout=["eq", "NCDHW"], uid=None, post_trace=True)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, include_padding_in_calculation=bool,
        layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def average_pooling3d(self, kernel_size=2, stride=None, padding=0, dilation=1, 
        return_indices=False, ceil_mode=False, include_padding_in_calculation=True, 
        layout='NCDHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "average_pooling3d";
        tmp["params"] = {};
        tmp["params"]["kernel_size"] = kernel_size;
        tmp["params"]["stride"] = stride;
        tmp["params"]["padding"] = padding;
        tmp["params"]["dilation"] = dilation;
        tmp["params"]["return_indices"] = return_indices;
        tmp["params"]["ceil_mode"] = ceil_mode;
        tmp["params"]["include_padding_in_calculation"] = include_padding_in_calculation;
        tmp["params"]["layout"] = layout;  
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=True)
    @error_checks(None, layout=["eq", "NCW"], uid=None, post_trace=True)
    @accepts("self", layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def global_max_pooling1d(self, layout='NCW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_max_pooling1d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=True)
    @error_checks(None, layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def global_max_pooling2d(self, layout='NCHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_max_pooling2d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout;
        return tmp; 
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=True)
    @error_checks(None, layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def global_max_pooling(self, layout='NCHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_max_pooling2d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout; 
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=True)
    @error_checks(None, layout=["eq", "NCDHW"], uid=None, post_trace=True)
    @accepts("self", layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def global_max_pooling3d(self, layout='NCDHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_max_pooling3d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout; 
        return tmp;
    #####################################################################################################################################

    


    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=True)
    @error_checks(None, layout=["eq", "NCW"], uid=None, post_trace=True)
    @accepts("self", layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def global_average_pooling1d(self, layout='NCW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_average_pooling1d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=True)
    @error_checks(None, layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def global_average_pooling2d(self, layout='NCHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_average_pooling2d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout; 
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=True)
    @error_checks(None, layout=["eq", "NCHW"], uid=None, post_trace=True)
    @accepts("self", layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def global_average_pooling(self, layout='NCHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_average_pooling2d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout; 
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=True)
    @error_checks(None, layout=["eq", "NCDHW"], uid=None, post_trace=True)
    @accepts("self", layout=str, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def global_average_pooling3d(self, layout='NCDHW', uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_average_pooling3d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout; 
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, units=None, use_bias=None, flatten=None, uid=None, post_trace=True)
    @error_checks(None, units=["gt", 0], use_bias=None, flatten=None, uid=None, post_trace=True)
    @accepts("self", units=int, use_bias=bool, flatten=bool, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def fully_connected(self, units=512, use_bias=True, flatten=True, uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "fully_connected";
        tmp["params"] = {};
        tmp["params"]["units"] = units; 
        tmp["params"]["use_bias"] = use_bias; 
        tmp["params"]["flatten"] = flatten; 
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, drop_probability=["lt", 0.5], use_bias=None, flatten=None, uid=None, post_trace=True)    
    @error_checks(None, drop_probability=["gte", 0.0, "lt", 1.0], use_bias=None, flatten=None, uid=None, post_trace=True)
    @accepts("self", drop_probability=[int, float], axes=tuple, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def dropout(self, drop_probability=0.2, axes=(), uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "dropout";
        tmp["params"] = {};
        tmp["params"]["drop_probability"] = drop_probability;
        tmp["params"]["axes"] = axes;
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def flatten(self, uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "flatten";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def identity(self, uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "identity";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, moving_average_momentum=["lt", 1.0], epsilon=["lt", 0.0001], use_trainable_parameters=None,
        activate_scale_shift_operation=None, uid=None, post_trace=True)
    @error_checks(None, moving_average_momentum=["gte", 0], epsilon=["gte", 0], use_trainable_parameters=None,
        activate_scale_shift_operation=None, uid=None, post_trace=True)
    @accepts("self", moving_average_momentum=[float, int], epsilon=float, use_trainable_parameters=bool,
        activate_scale_shift_operation=bool, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def batch_normalization(self, moving_average_momentum=0.1, epsilon=0.00001, use_trainable_parameters=True, 
        activate_scale_shift_operation=False, uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "batch_normalization";
        tmp["params"] = {};
        tmp["params"]["moving_average_momentum"] = moving_average_momentum;
        tmp["params"]["epsilon"] = epsilon;
        tmp["params"]["use_trainable_parameters"] = use_trainable_parameters;
        tmp["params"]["activate_scale_shift_operation"] = activate_scale_shift_operation;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, moving_average_momentum=["lt", 1.0], epsilon=["lt", 0.0001], use_trainable_parameters=None,
        uid=None, post_trace=True)
    @error_checks(None, moving_average_momentum=["gte", 0], epsilon=["gte", 0], use_trainable_parameters=None,
        uid=None, post_trace=True)
    @accepts("self", moving_average_momentum=[float, int], epsilon=float, use_trainable_parameters=bool,
        uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def instance_normalization(self, moving_average_momentum=0.1, epsilon=0.00001, use_trainable_parameters=False, 
        uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "instance_normalization";
        tmp["params"] = {};
        tmp["params"]["moving_average_momentum"] = moving_average_momentum;
        tmp["params"]["epsilon"] = epsilon;
        tmp["params"]["use_trainable_parameters"] = use_trainable_parameters;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, moving_average_momentum=["lt", 1.0], epsilon=["lt", 0.0001], use_trainable_parameters=None,
        uid=None, post_trace=True)
    @error_checks(None, moving_average_momentum=["gte", 0], epsilon=["gte", 0], use_trainable_parameters=None,
        uid=None, post_trace=True)
    @accepts("self", moving_average_momentum=[float, int], epsilon=float, use_trainable_parameters=bool,
        uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def layer_normalization(self, moving_average_momentum=0.9, epsilon=0.00001, use_trainable_parameters=True, 
        uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "layer_normalization";
        tmp["params"] = {};
        tmp["params"]["epsilon"] = epsilon;
        tmp["params"]["use_trainable_parameters"] = use_trainable_parameters;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def add(self, uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "add";
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def concatenate(self, uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "concatenate";
        return tmp;
    #####################################################################################################################################





    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def relu(self, uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "relu";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def sigmoid(self, uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "sigmoid";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def tanh(self, uid=None):
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "tanh";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, beta=None, threshold=None, uid=None, post_trace=True)
    @error_checks(None, beta=["gt", 0], threshold=None, uid=None, post_trace=True)
    @accepts("self", beta=[int, float], threshold=[int, float], uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def softplus(self, beta=1, threshold=20, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "softplus";
        tmp["params"] = {};
        tmp["params"]["beta"] = beta;
        tmp["params"]["threshold"] = threshold;
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def softsign(self, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "softsign";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, alpha=None, uid=None, post_trace=True)
    @error_checks(None, alpha=["gt", 0], uid=None, post_trace=True)
    @accepts("self", alpha=float, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def elu(self, alpha=1.0, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "elu";
        tmp["params"] = {};
        tmp["params"]["alpha"] = alpha;
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, alpha=None, uid=None, post_trace=True)
    @error_checks(None, alpha=["gt", 0], uid=None, post_trace=True)
    @accepts("self", alpha=float, uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def leaky_relu(self, alpha=0.01, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "leaky_relu";
        tmp["params"] = {};
        tmp["params"]["alpha"] = alpha;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def prelu(self, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "prelu";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def selu(self, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "selu";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, threshold=None, uid=None, post_trace=True)
    @error_checks(None, threshold=None, uid=None, post_trace=True)
    @accepts("self", threshold=[int, float], uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def hardshrink(self, threshold=0.5, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "hardshrink";
        tmp["params"] = {};
        tmp["params"]["threshold"] = threshold;
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, min_val=None, max_val=None, uid=None, post_trace=True)
    @error_checks(None, min_val=None, max_val=None, uid=None, post_trace=True)
    @accepts("self", min_val=[int, float], max_val=[int, float], uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def hardtanh(self, min_val=-1.0, max_val=1.0, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "hardtanh";
        tmp["params"] = {};
        tmp["params"]["min_val"] = min_val;
        tmp["params"]["max_val"] = max_val;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def logsigmoid(self, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "logsigmoid";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def relu6(self, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "relu6";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, lower=None, upper=None, uid=None, post_trace=True)
    @error_checks(None, lower=None, upper=None, uid=None, post_trace=True)
    @accepts("self", lower=[int, float], upper=[int, float], uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def rrelu(self, lower=0.125, upper=0.333, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "rrelu";
        tmp["params"] = {};
        tmp["params"]["lower"] = lower;
        tmp["params"]["upper"] = upper;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, alpha=None, uid=None, post_trace=True)
    @error_checks(None, alpha=None, uid=None, post_trace=True)
    @accepts("self", alpha=[int, float], uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def celu(self, alpha=1.0, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "celu";
        tmp["params"] = {};
        tmp["params"]["alpha"] = alpha;
        return tmp;
    #####################################################################################################################################    


    #####################################################################################################################################
    @warning_checks(None, threshold=None, uid=None, post_trace=True)
    @error_checks(None, threshold=None, uid=None, post_trace=True)
    @accepts("self", threshold=[int, float], uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def softshrink(self, threshold=0.5, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "softshrink";
        tmp["params"] = {};
        tmp["params"]["threshold"] = threshold;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def tanhshrink(self, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "tanhshrink";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, threshold=None, value=None, uid=None, post_trace=True)
    @error_checks(None, threshold=None, value=None, uid=None, post_trace=True)
    @accepts("self", threshold=[int, float], value=[int, float], uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def threshold(self, threshold=1.0, value=0.01, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "threshold";
        tmp["params"] = {};
        tmp["params"]["threshold"] = threshold;
        tmp["params"]["value"] = value;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def softmin(self, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "softmin";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def softmax(self, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "softmax";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=True)
    @error_checks(None, uid=None, post_trace=True)
    @accepts("self", uid=[None, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def logsoftmax(self, uid=None): 
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "logsoftmax";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################







