import math
from monk.tf_keras_1.finetune.imports import *
from monk.system.imports import *

from monk.tf_keras_1.finetune.level_7_aux_main import prototype_aux


class prototype_layers(prototype_aux):
    @accepts("self", verbose=int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    #####################################################################################################################################
    @warning_checks(None, num_neurons=["lt", 10000], final_layer=None, post_trace=False)
    @error_checks(None, num_neurons=["gt", 0], final_layer=None, post_trace=False)
    @accepts("self", num_neurons=[int, bool], final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_linear(self, num_neurons=False, final_layer=False):
        '''
        Append dense (fully connected) layer to base network in transfer learning

        Args:
            num_neurons (int): Number of neurons in the dense layer
            final_layer (bool): If True, then number of neurons are directly set as number of classes in dataset for single label type classification

        Returns:
            None
        '''
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
    @warning_checks(None, probability=["lt", 0.7], final_layer=None, post_trace=False)
    @error_checks(None, probability=["gt", 0, "lt", 1], final_layer=None, post_trace=False)
    @accepts("self", probability=float, final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_dropout(self, probability=0.5, final_layer=False):
        '''
        Append dropout layer to base network in transfer learning

        Args:
            probability (float): Droping probability of neurons in next layer
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = layer_dropout(self.system_dict, probability=probability, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, final_layer=None, post_trace=False)
    @error_checks(None, final_layer=None, post_trace=False)
    @accepts("self", final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_globalaveragepooling(self, final_layer=False):
        '''
        Append global average pooling layer to base network in transfer learning

        Args:
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = layer_globalaveragepooling(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, final_layer=None, post_trace=False)
    @error_checks(None, final_layer=None, post_trace=False)
    @accepts("self", final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_flatten(self, final_layer=False):
        '''
        Append flatten layer to base network in transfer learning

        Args:
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = layer_flatten(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, negative_slope=["lt", 0.2], final_layer=None, post_trace=False)
    @error_checks(None, negative_slope=["gt", 0], final_layer=None, post_trace=False)
    @accepts("self", negative_slope=[float, int], final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_leakyrelu(self, negative_slope=0.01, final_layer=False):
        '''
        Append Leaky - ReLU activation to base network in transfer learning

        Args:
            negative_slope (float): Multiplicatve factor towards negative spectrum of real numbers.
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_leakyrelu(self.system_dict, negative_slope=negative_slope, final_layer=final_layer);
    #####################################################################################################################################





    #####################################################################################################################################
    @warning_checks(None, num_parameters=["lt", 1], final_layer=None, post_trace=False)
    @error_checks(None, num_parameters=["gt", 0], init=["gt", 0], final_layer=None, post_trace=False)
    @accepts("self", num_parameters=int, init=[int, float], final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_prelu(self, num_parameters=1, init=0.25, final_layer=False):
        '''
        Append Learnable parameerized rectified linear unit activation to base network in transfer learning

        Args:
            init (float): Initialization value for multiplicatve factor towards negative spectrum of real numbers.
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_prelu(self.system_dict, num_parameters=num_parameters, init=init, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @accepts("self", alpha=[float, int], final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_elu(self, alpha=1.0, final_layer=False):
        '''
        Append exponential linear unit activation to base network in transfer learning

        Args:
            alpha (float): Multiplicatve factor.
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_elu(self.system_dict, alpha=alpha, final_layer=final_layer); 
    #####################################################################################################################################






    #####################################################################################################################################
    @accepts("self", [int, float], [int, float], final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_threshold(self, threshold, value, final_layer=False):
        '''
        Append threshold activation to base network in transfer learning

        Args:
            threshold (float): thereshold limit value
            value (float): replacement value if input is off-limits
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_threshold(self.system_dict, threshold, value, final_layer=final_layer);
    #####################################################################################################################################





    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_softmax(self, final_layer=False):
        '''
        Append softmax activation to base network in transfer learning

        Args:
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_softmax(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################





    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_relu(self, final_layer=False):
        '''
        Append rectified linear unit activation to base network in transfer learning

        Args:
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_relu(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################





    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_selu(self, final_layer=False):
        '''
        Append scaled exponential linear unit activation to base network in transfer learning

        Args:
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_selu(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################






    #####################################################################################################################################
    @accepts("self", beta=[int, float], threshold=[int, float], final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_softplus(self, beta=1, threshold=20, final_layer=False):
        '''
        Append softplus activation to base network in transfer learning

        Args:
            threshold (int): softplus (thresholded relu) limit 
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_softplus(self.system_dict, beta=beta, threshold=threshold, final_layer=final_layer);
    #####################################################################################################################################






    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_softsign(self, final_layer=False):
        '''
        Append softsign activation to base network in transfer learning

        Args:
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_softsign(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################






    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_tanh(self, final_layer=False):
        '''
        Append tanh activation to base network in transfer learning

        Args:
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_tanh(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################






    #####################################################################################################################################
    @accepts("self", final_layer=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def append_sigmoid(self, final_layer=False):
        '''
        Append rectified linear unit activation to base network in transfer learning

        Args:
            final_layer (bool): Indicator that this layer marks the end of network.

        Returns:
            None
        '''
        if(self.system_dict["model"]["final_layer"]):
            msg = "Cannot append more layers.\n";
            msg += "Tip: Previously appended layer termed as final layer";
            raise ConstraintError(msg);
        else:
            self.system_dict = activation_sigmoid(self.system_dict, final_layer=final_layer);
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, output_channels=["lt", 2048], kernel_size=["lt", 16], stride=["lt", 16], padding=None,
        groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCW", "eq", "NWC"], uid=None, post_trace=False)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        groups=int, dilation=int, use_bias=bool, layout=str, uid=[type(None), str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def convolution1d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        groups=1, dilation=1, use_bias=True, layout='NWC', uid=None):
        '''
        Append 1d-convolution to custom network

        Args:
            output_channels (int): Number of output features for this layers
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            groups (int): Number of groups for grouped convolution
            dilation (int): Factor for dilated convolution
            use_bias (bool): If True, learnable bias is added
            layout (str): Either of these values (order)
                            1) 'NCW' - order
                            2) 'NWC' - order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", output_channels=int, kernel_size=[int, tuple], stride=[int, tuple], padding=[str, int, tuple],
        groups=int, dilation=int, use_bias=bool, layout=str, uid=[type(None), str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def convolution2d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        groups=1, dilation=1, use_bias=True, layout='NHWC', uid=None):
        '''
        Append 2d-convolution to custom network

        Args:
            output_channels (int): Number of output features for this layers
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            groups (int): Number of groups for grouped convolution
            dilation (int): Factor for dilated convolution
            use_bias (bool): If True, learnable bias is added
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", output_channels=int, kernel_size=[int, tuple], stride=[int, tuple], padding=[str, int, tuple],
        groups=int, dilation=int, use_bias=bool, layout=str, uid=[type(None), str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def convolution(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        groups=1, dilation=1, use_bias=True, layout='NHWC', uid=None):
        '''
        Append 2d-convolution to custom network

        Args:
            output_channels (int): Number of output features for this layers
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            groups (int): Number of groups for grouped convolution
            dilation (int): Factor for dilated convolution
            use_bias (bool): If True, learnable bias is added
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCDHW", "eq", "NDHWC"], uid=None, post_trace=False)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        groups=int, dilation=int, use_bias=bool, layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def convolution3d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        groups=1, dilation=1, use_bias=True, layout='NDHWC', uid=None):
        '''
        Append 3d-convolution to custom network

        Args:
            output_channels (int): Number of output features for this layers
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            output_padding (int): Additional padding applied to output
            groups (int): Number of groups for grouped convolution
            dilation (int): Factor for dilated convolution
            use_bias (bool): If True, learnable bias is added
            layout (str): Either of these values (order)
                            1) 'NCDHW' - Order
                            2) 'NDHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - D: Depth of features in layers
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        output_padding=None, groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        output_padding=["gte, 0"], groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        output_padding=int, groups=int, dilation=int, use_bias=bool, layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def transposed_convolution(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        output_padding=0, groups=1, dilation=1, use_bias=True, layout='NHWC', uid=None):
        '''
        Append 2d-transposed-convolution to custom network

        Args:
            output_channels (int): Number of output features for this layers
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            output_padding (int): Additional padding applied to output
            groups (int): Number of groups for grouped convolution
            dilation (int): Factor for dilated convolution
            use_bias (bool): If True, learnable bias is added
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        output_padding=None, groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        output_padding=["gte, 0"], groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        output_padding=int, groups=int, dilation=int, use_bias=bool, layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def transposed_convolution2d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        output_padding=0, groups=1, dilation=1, use_bias=True, layout='NHWC', uid=None):
        '''
        Append 2d-transposed-convolution to custom network

        Args:
            output_channels (int): Number of output features for this layers
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            output_padding (int): Additional padding applied to output
            groups (int): Number of groups for grouped convolution
            dilation (int): Factor for dilated convolution
            use_bias (bool): If True, learnable bias is added
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        output_padding=None, groups=None, dilation=None, use_bias=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], kernel_size=["gt", 0], stride=["gte", 1], padding=None,
        output_padding=["gte, 0"], groups=["gte", 1], dilation=["gte", 1], use_bias=None, layout=["eq", "NCDHW", "eq", "NDHWC"], uid=None, post_trace=False)
    @accepts("self", output_channels=int, kernel_size=int, stride=int, padding=[str, int],
        output_padding=int, groups=int, dilation=int, use_bias=bool, layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def transposed_convolution3d(self, output_channels=3, kernel_size=3, stride=1, padding="in_eq_out", 
        output_padding=0, groups=1, dilation=1, use_bias=True, layout='NDHWC', uid=None):
        '''
        Append 3d-transposed-convolution to custom network

        Args:
            output_channels (int): Number of output features for this layers
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            output_padding (int): Additional padding applied to output
            groups (int): Number of groups for grouped convolution
            dilation (int): Factor for dilated convolution
            use_bias (bool): If True, learnable bias is added
            layout (str): Either of these values (order)
                            1) 'NCDHW' - Order
                            2) 'NDHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - D: Depth of features in layers
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        return_indices=None, ceil_mode=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, layout=["eq", "NCW", "eq", "NWC"], uid=None, post_trace=False)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def max_pooling1d(self, kernel_size=2, stride=None, padding="in_eq_out", dilation=1, 
        return_indices=False, ceil_mode=False, layout='NWC', uid=None):
        '''
        Append 1d-max-pooling to custom network

        Args:
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            dilation (int): Factor for dilated pooling
            return_indices (bool): Fixed value set as False
            ceil_mode (bool): If True, apply ceil math operation post pooling
            layout (str): Either of these values (order)
                            1) 'NCW' - order
                            2) 'NWC' - order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        return_indices=None, ceil_mode=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int, 
        return_indices=bool, ceil_mode=bool, layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def max_pooling2d(self, kernel_size=2, stride=None, padding="in_eq_out", dilation=1, 
        return_indices=False, ceil_mode=False, layout='NHWC', uid=None):
        '''
        Append 2d-max-pooling to custom network

        Args:
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            dilation (int): Factor for dilated pooling
            return_indices (bool): Fixed value set as False
            ceil_mode (bool): If True, apply ceil math operation post pooling
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        return_indices=None, ceil_mode=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def max_pooling(self, kernel_size=2, stride=None, padding="in_eq_out", dilation=1, 
        return_indices=False, ceil_mode=False, layout='NHWC', uid=None):
        '''
        Append 2d-max-pooling to custom network

        Args:
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            dilation (int): Factor for dilated pooling
            return_indices (bool): Fixed value set as False
            ceil_mode (bool): If True, apply ceil math operation post pooling
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        return_indices=None, ceil_mode=None, layout=None, uid=None, post_trace=False)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, layout=["eq", "NCDHW", "eq", "NDHWC"], uid=None, post_trace=False)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def max_pooling3d(self, kernel_size=2, stride=None, padding="in_eq_out", dilation=1, 
        return_indices=False, ceil_mode=False, layout='NDHWC', uid=None):
        '''
        Append 3d-max-pooling to custom network

        Args:
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            dilation (int): Factor for dilated pooling
            return_indices (bool): Fixed value set as False
            ceil_mode (bool): If True, apply ceil math operation post pooling
            layout (str): Either of these values (order)
                            1) 'NCDHW' - Order
                            2) 'NDHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - D: Depth of features in layers
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        layout=None, uid=None, post_trace=False)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None,
        layout=["eq", "NCW", "eq", "NWC"], uid=None, post_trace=False)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, include_padding_in_calculation=bool,
        layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def average_pooling1d(self, kernel_size=2, stride=None, padding="in_eq_out", dilation=1, 
        return_indices=False, ceil_mode=False, include_padding_in_calculation=True, 
        layout='NWC', uid=None):
        '''
        Append 1d-average-pooling to custom network

        Args:
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            dilation (int): Factor for dilated pooling
            return_indices (bool): Fixed value set as False
            include_padding_in_calculation (bool): If True, padding will be considered.
            ceil_mode (bool): If True, apply ceil math operation post pooling
            layout (str): Either of these values (order)
                            1) 'NCW' - order
                            2) 'NWC' - order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        layout=None, uid=None, post_trace=False)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None,
        layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, include_padding_in_calculation=bool,
        layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def average_pooling2d(self, kernel_size=2, stride=None, padding="in_eq_out", dilation=1, 
        return_indices=False, ceil_mode=False, include_padding_in_calculation=True, 
        layout='NHWC', uid=None):
        '''
        Append 2d-average-pooling to custom network

        Args:
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            dilation (int): Factor for dilated pooling
            return_indices (bool): Fixed value set as False
            ceil_mode (bool): If True, apply ceil math operation post pooling
            include_padding_in_calculation (bool): If True, padding will be considered.
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        layout=None, uid=None, post_trace=False)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None,
        layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, include_padding_in_calculation=bool,
        layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def average_pooling(self, kernel_size=2, stride=None, padding="in_eq_out", dilation=1, 
        return_indices=False, ceil_mode=False, include_padding_in_calculation=True, 
        layout='NHWC', uid=None):
        '''
        Append 2d-average-pooling to custom network

        Args:
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            dilation (int): Factor for dilated pooling
            return_indices (bool): Fixed value set as False
            ceil_mode (bool): If True, apply ceil math operation post pooling
            include_padding_in_calculation (bool): If True, padding will be considered.
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
        layout=None, uid=None, post_trace=False)
    @error_checks(None, kernel_size=["gt", 0], stride=["gte", 1], padding=None, dilation=["gte", 1],
        return_indices=None, ceil_mode=None, include_padding_in_calculation=None,
        layout=["eq", "NCDHW", "eq", "NDHWC"], uid=None, post_trace=False)
    @accepts("self", kernel_size=int, stride=[int, None], padding=[str, int], dilation=int,
        return_indices=bool, ceil_mode=bool, include_padding_in_calculation=bool,
        layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def average_pooling3d(self, kernel_size=2, stride=None, padding="in_eq_out", dilation=1, 
        return_indices=False, ceil_mode=False, include_padding_in_calculation=True, 
        layout='NDHWC', uid=None):
        '''
        Append 3d-average-pooling to custom network

        Args:
            kernel_size (int, tuple): kernel matrix size 
            stride (int, tuple): kernel movement stride  
            padding (int, tuple, str): Zero padding applied to input
                                        1) "in_eq_out": Automated padding applied to keep output shape same as input
                                        2) integer or tuple value: Manually add padding 
            dilation (int): Factor for dilated pooling
            return_indices (bool): Fixed value set as False
            ceil_mode (bool): If True, apply ceil math operation post pooling
            include_padding_in_calculation (bool): If True, padding will be considered.
            layout (str): Either of these values (order)
                            1) 'NCDHW' - Order
                            2) 'NDHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - D: Depth of features in layers
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
    @warning_checks(None, layout=None, uid=None, post_trace=False)
    @error_checks(None, layout=["eq", "NCW", "eq", "NWC"], uid=None, post_trace=False)
    @accepts("self", layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def global_max_pooling1d(self, layout='NWC', uid=None):
        '''
        Append 1d-global-max-pooling to custom network

        Args:
            layout (str): Either of these values (order)
                            1) 'NCW' - order
                            2) 'NWC' - order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_max_pooling1d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=False)
    @error_checks(None, layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def global_max_pooling2d(self, layout='NHWC', uid=None):
        '''
        Append 2d-global-max-pooling to custom network

        Args:
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_max_pooling2d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout;
        return tmp; 
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=False)
    @error_checks(None, layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def global_max_pooling(self, layout='NHWC', uid=None):
        '''
        Append 2d-global-max-pooling to custom network

        Args:
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_max_pooling2d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout; 
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=False)
    @error_checks(None, layout=["eq", "NCDHW", "eq", "NDHWC"], uid=None, post_trace=False)
    @accepts("self", layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def global_max_pooling3d(self, layout='NDHWC', uid=None):
        '''
        Append 3d-global-max-pooling to custom network

        Args:
            layout (str): Either of these values (order)
                            1) 'NCDHW' - Order
                            2) 'NDHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - D: Depth of features in layers
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_max_pooling3d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout; 
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=False)
    @error_checks(None, layout=["eq", "NCW", "eq", "NWC"], uid=None, post_trace=False)
    @accepts("self", layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def global_average_pooling1d(self, layout='NWC', uid=None):
        '''
        Append 1d-global-average-pooling to custom network

        Args:
            layout (str): Either of these values (order)
                            1) 'NCW' - order
                            2) 'NWC' - order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_average_pooling1d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=False)
    @error_checks(None, layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def global_average_pooling2d(self, layout='NHWC', uid=None):
        '''
        Append 2d-global-average-pooling to custom network

        Args:
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_average_pooling2d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout; 
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=False)
    @error_checks(None, layout=["eq", "NCHW", "eq", "NHWC"], uid=None, post_trace=False)
    @accepts("self", layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def global_average_pooling(self, layout='NHWC', uid=None):
        '''
        Append 2d-global-average-pooling to custom network

        Args:
            layout (str): Either of these values (order)
                            1) 'NCHW' - Order
                            2) 'NHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_average_pooling2d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout; 
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=False)
    @error_checks(None, layout=["eq", "NCDHW", "eq", "NDHWC"], uid=None, post_trace=False)
    @accepts("self", layout=str, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def global_average_pooling3d(self, layout='NDHWC', uid=None):
        '''
        Append 3d-global-average-pooling to custom network

        Args:
            layout (str): Either of these values (order)
                            1) 'NCDHW' - Order
                            2) 'NDHWC' - Order
                            - N: Number of elements in batches
                            - C: Number of channels
                            - D: Depth of features in layers
                            - H: Height of features in layers
                            - W: Number of features in layers
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "global_average_pooling3d";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout; 
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, layout=None, uid=None, post_trace=False)
    @error_checks(None, layout=None, uid=None, post_trace=False)
    @accepts("self", layout=["eq", "NCHW", "eq", "NHWC"], uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def flatten(self, layout='NHWC', uid=None):
        '''
        Append flatten layer to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "flatten";
        tmp["params"] = {};
        tmp["params"]["layout"] = layout;
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, units=None, use_bias=None, flatten=None, uid=None, post_trace=False)
    @error_checks(None, units=["gt", 0], use_bias=None, flatten=None, uid=None, post_trace=False)
    @accepts("self", units=int, use_bias=bool, flatten=bool, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def fully_connected(self, units=512, use_bias=True, flatten=True, uid=None):
        '''
        Append fully-connected (dense) layer to custom network

        Args:
            units (int): Number of neurons in the layer
            use_bias (bool): If True, learnable bias is added
            flatten (bool): Fixed to True
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
    @warning_checks(None, drop_probability=["lt", 0.5], use_bias=None, flatten=None, uid=None, post_trace=False)    
    @error_checks(None, drop_probability=["gte", 0.0, "lt", 1.0], use_bias=None, flatten=None, uid=None, post_trace=False)
    @accepts("self", drop_probability=[int, float], axes=tuple, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def dropout(self, drop_probability=0.2, axes=(), uid=None):
        '''
        Append dropout layer to custom network

        Args:
            drop_probability (float): Probability for not considering neurons in the output
            axes (tuple): Channel axis to implement dropout over
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "dropout";
        tmp["params"] = {};
        tmp["params"]["drop_probability"] = drop_probability;
        tmp["params"]["axes"] = axes;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def identity(self, uid=None):
        '''
        Append identity layer to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "identity";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, moving_average_momentum=["lt", 1.0], epsilon=["lt", 0.0001], use_trainable_parameters=None,
        activate_scale_shift_operation=None, uid=None, post_trace=False)
    @error_checks(None, moving_average_momentum=["gte", 0], epsilon=["gte", 0], use_trainable_parameters=None,
        activate_scale_shift_operation=None, uid=None, post_trace=False)
    @accepts("self", moving_average_momentum=[float, int], epsilon=float, use_trainable_parameters=bool,
        activate_scale_shift_operation=bool, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def batch_normalization(self, moving_average_momentum=0.99, epsilon=0.001, use_trainable_parameters=True, 
        activate_scale_shift_operation=False, uid=None):
        '''
        Append batch normalization layer to custom network

        Args:
            moving_average_momentum (float): Normalization momentum value
            epsilon (float): Value to avoid division by zero
            use_trainable_paramemetrs (bool): If True, batch norm turns into a trainable layer
            activate_scale_shift_operation (bool): Fixed status - False
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
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
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def add(self, uid=None):
        '''
        Append elementwise addition layer to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "add";
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def concatenate(self, uid=None):
        '''
        Append concatenation layer to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "concatenate";
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def relu(self, uid=None):
        '''
        Append rectified linear unit activation to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "relu";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, axis=None, uid=None, post_trace=False)
    @error_checks(None, axis=None, uid=None, post_trace=False)
    @accepts("self", axis=int, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def softmax(self, axis=-1, uid=None):
        '''
        Append softmax activation to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "softmax";
        tmp["params"] = {};
        tmp["params"]["axis"] = axis;
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, threshold=None, uid=None, post_trace=False)
    @error_checks(None, threshold=None, uid=None, post_trace=False)
    @accepts("self", threshold=int, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def thresholded_relu(self, threshold=1.0, uid=None):
        '''
        Append thresholded relu activation to custom network

        Args:
            threshold (float): Thresh limits
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "thresholded_relu";
        tmp["params"] = {};
        tmp["params"]["threshold"] = threshold;
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, alpha=None, uid=None, post_trace=False)
    @error_checks(None, alpha=["gt", 0], uid=None, post_trace=False)
    @accepts("self", alpha=float, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def elu(self, alpha=1.0, uid=None): 
        '''
        Append exponential linear unit activation to custom network

        Args:
            alpha (float): Multiplicative factor
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "elu";
        tmp["params"] = {};
        tmp["params"]["alpha"] = alpha;
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def prelu(self, uid=None): 
        '''
        Append paramemeterized rectified linear unit activation to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "prelu";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, alpha=None, uid=None, post_trace=False)
    @error_checks(None, alpha=["gt", 0], uid=None, post_trace=False)
    @accepts("self", alpha=float, uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def leaky_relu(self, alpha=0.3, uid=None): 
        '''
        Append leaky relu activation to custom network

        Args:
            alpha (float): Multiplicatve factor towards negative spectrum of real numbers.
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "leaky_relu";
        tmp["params"] = {};
        tmp["params"]["alpha"] = alpha;
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def selu(self, uid=None):
        '''
        Append scaled exponential linear unit activation to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "selu";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def softplus(self, uid=None):
        '''
        Append softplus activation to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "softplus";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def softsign(self, uid=None):
        '''
        Append softsign activation to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "softsign";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def tanh(self, uid=None):
        '''
        Append tanh activation to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "tanh";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def sigmoid(self, uid=None):
        '''
        Append sigmoid activation to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "sigmoid";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################


    #####################################################################################################################################
    @warning_checks(None, uid=None, post_trace=False)
    @error_checks(None, uid=None, post_trace=False)
    @accepts("self", uid=[None, str], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def hard_sigmoid(self, uid=None):
        '''
        Append Hard-Sigmoid activation to custom network

        Args:
            uid (str): Unique name for layer, if not mentioned then dynamically assigned

        Returns:
            dict: Containing all the parameters set as per function arguments
        '''
        tmp = {};
        tmp["uid"] = uid;
        tmp["name"] = "hard_sigmoid";
        tmp["params"] = {};
        return tmp;
    #####################################################################################################################################





    
    #####################################################################################################################################
    @warning_checks(None, output_channels=None, stride=None, downsample=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], stride=None, downsample=None, post_trace=False)
    @accepts("self", output_channels=int, stride=[None, int, tuple], downsample=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def resnet_v1_block(self, output_channels=16, stride=1, downsample=True):
        '''
        Append Resnet V1 Block to custom network

        Args:
            output_channels (int): Number of output features for this block
            stride (int, tuple): kernel movement stride  
            downsample (bool): If False, residual branch is a shortcut,
                                Else, residual branch has non-identity layers

        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
    
        subnetwork = [];
        branch_1 = [];
        branch_1.append(self.convolution(output_channels=output_channels, kernel_size=3, stride=stride));
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=output_channels, kernel_size=3, stride=1));
        branch_1.append(self.batch_normalization());
        
        branch_2 = [];
        if(downsample):
            branch_2.append(self.convolution(output_channels=output_channels, kernel_size=1, stride=stride));
            branch_2.append(self.batch_normalization());
        else:
            branch_2.append(self.identity());
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(self.add());
        
        network.append(subnetwork)
        network.append(self.relu());
        return network;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, output_channels=None, stride=None, downsample=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], stride=None, downsample=None, post_trace=False)
    @accepts("self", output_channels=int, stride=[None, int, tuple], downsample=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def resnet_v2_block(self, output_channels=16, stride=1, downsample=True):
        '''
        Append Resnet V2 Block to custom network

        Args:
            output_channels (int): Number of output features for this block
            stride (int, tuple): kernel movement stride  
            downsample (bool): If False, residual branch is a shortcut,
                                Else, residual branch has non-identity layers

        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        network.append(self.batch_normalization());
        network.append(self.relu());
        
        subnetwork = [];
        branch_1 = [];
        branch_1.append(self.convolution(output_channels=output_channels, kernel_size=3, stride=stride));
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=output_channels, kernel_size=3, stride=1));
        
        branch_2 = [];
        if(downsample):
            branch_2.append(self.convolution(output_channels=output_channels, kernel_size=1, stride=stride));
        else:
            branch_2.append(self.identity());
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(self.add());
        
        network.append(subnetwork);
        return network;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, output_channels=None, stride=None, downsample=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], stride=None, downsample=None, post_trace=False)
    @accepts("self", output_channels=int, stride=[None, int, tuple], downsample=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def resnet_v1_bottleneck_block(self, output_channels=16, stride=1, downsample=True):
        '''
        Append Resnet V1 Bottleneck Block to custom network

        Args:
            output_channels (int): Number of output features for this block
            stride (int, tuple): kernel movement stride  
            downsample (bool): If False, residual branch is a shortcut,
                                Else, residual branch has non-identity layers

        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
    
        subnetwork = [];
        branch_1 = [];
        branch_1.append(self.convolution(output_channels=output_channels//4, kernel_size=1, stride=stride));
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=output_channels//4, kernel_size=3, stride=1));
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=output_channels, kernel_size=1, stride=1));
        branch_1.append(self.batch_normalization());
        
        branch_2 = [];
        if(downsample):
            branch_2.append(self.convolution(output_channels=output_channels, kernel_size=1, stride=stride));
            branch_2.append(self.batch_normalization());
        else:
            branch_2.append(self.identity());
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(self.add());
        
        network.append(subnetwork)
        network.append(self.relu())
        return network;
    #####################################################################################################################################





    #####################################################################################################################################
    @warning_checks(None, output_channels=None, stride=None, downsample=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], stride=None, downsample=None, post_trace=False)
    @accepts("self", output_channels=int, stride=[None, int, tuple], downsample=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def resnet_v2_bottleneck_block(self, output_channels=16, stride=1, downsample=True):
        '''
        Append Resnet V2 Bottleneck Block to custom network

        Args:
            output_channels (int): Number of output features for this block
            stride (int, tuple): kernel movement stride  
            downsample (bool): If False, residual branch is a shortcut,
                                Else, residual branch has non-identity layers

        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        network.append(self.batch_normalization());
        network.append(self.relu());
        
        subnetwork = [];
        branch_1 = [];
        branch_1.append(self.convolution(output_channels=output_channels//4, kernel_size=1, stride=1));
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=output_channels//4, kernel_size=3, stride=stride));
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=output_channels, kernel_size=1, stride=1));
        
        branch_2 = [];
        if(downsample):
            branch_2.append(self.convolution(output_channels=output_channels, kernel_size=1, stride=stride));
        else:
            branch_2.append(self.identity());
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(self.add());
        
        network.append(subnetwork)
        return network;
    #####################################################################################################################################





    #####################################################################################################################################
    @warning_checks(None, output_channels=None, cardinality=None, bottleneck_width=None, stride=None, 
        downsample=None, post_trace=False)
    @error_checks(None, output_channels=["gt", 0], cardinality=None, bottleneck_width=None, stride=None, 
        downsample=None, post_trace=False)
    @accepts("self", output_channels=int, cardinality=int, bottleneck_width=int, stride=[int, tuple], 
        downsample=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def resnext_block(self, output_channels=256, cardinality=8, bottleneck_width=4, stride=1, downsample=True):
        '''
        Append Resnext Block to custom network

        Args:
            output_channels (int): Number of output features for this block
            cardinality (int): cardinality dimensions for complex transformations
            bottleneck_width (int): Bottleneck dimensions for reducing number of features
            stride (int): kernel movement stride  
            downsample (bool): If False, residual branch is a shortcut,
                                Else, residual branch has non-identity layers

        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
    
        channels = output_channels//4;
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D
        
        subnetwork = [];
        branch_1 = [];
        branch_1.append(self.convolution(output_channels=group_width, kernel_size=1, stride=1));
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=group_width, kernel_size=3, stride=stride));
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=output_channels, kernel_size=1, stride=1));
        branch_1.append(self.batch_normalization());
        
        
        
        branch_2 = [];
        if(downsample):
            branch_2.append(self.convolution(output_channels=output_channels, kernel_size=1, stride=stride));
            branch_2.append(self.batch_normalization());
        else:
            branch_2.append(self.identity());
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(self.add());
        
        network.append(subnetwork)
        network.append(self.relu());
        return network;
    #####################################################################################################################################





    #####################################################################################################################################
    @warning_checks(None, output_channels=None, bottleneck_width=None, stride=None, 
        post_trace=False)
    @error_checks(None, output_channels=["gt", 0], bottleneck_width=None, stride=None, 
        post_trace=False)
    @accepts("self", output_channels=int, bottleneck_width=int, stride=[int, tuple], 
        post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def mobilenet_v2_linear_bottleneck_block(self, output_channels=32, bottleneck_width=4, stride=1):
        '''
        Append Mobilenet V2 Linear Bottleneck Block to custom network

        Args:
            output_channels (int): Number of output features for this block
            stride (int): kernel movement stride  
            bottleneck_width (int): Bottleneck dimensions for reducing number of features

        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        
        subnetwork = [];
        branch_1 = [];
        if(bottleneck_width != 1):
            branch_1.append(self.convolution(output_channels=output_channels*bottleneck_width, 
                                            kernel_size=1, stride=1));
        
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=output_channels*bottleneck_width,
                                        kernel_size=3, stride=stride));
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=output_channels, kernel_size=1, stride=1));
        branch_1.append(self.batch_normalization());
        
        
        branch_2 = [];
        branch_2.append(self.identity());
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(self.add());
        
        network.append(subnetwork);
        return network;
    #####################################################################################################################################





    #####################################################################################################################################
    @warning_checks(None, input_channels=None, output_channels=None, kernel_size=None, stride=None, 
        padding=None, post_trace=False)
    @error_checks(None, input_channels=["gt", 0], output_channels=["gt", 0], kernel_size=None, stride=None, 
        padding=None, post_trace=False)
    @accepts("self", input_channels=int, output_channels=int, kernel_size=int, stride=[None, int, tuple], 
        padding=[None, int, tuple], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def separable_convolution_block(self, input_channels=16, output_channels=32, kernel_size=3, stride=1, padding=None):
        '''
        Append Separable convolution Block to custom network

        Args:
            input_channels (int): Number of input features for this block
            output_channels (int): Number of output features for this block
            kernel_size (int): Kernel matrix shape for all layers in this block
            stride (int): kernel movement stride  
            padding (int, tuple): external zero padding on input

        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        network.append(self.convolution(output_channels=input_channels, kernel_size=kernel_size, 
                                       stride=stride, groups=input_channels));
        network.append(self.convolution(output_channels=output_channels, kernel_size=1, 
                                       stride=1));

        return network;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, output_channels=None, bottleneck_width=None, stride=None, 
        post_trace=False)
    @error_checks(None, output_channels=["gt", 0], bottleneck_width=None, stride=None, 
        post_trace=False)
    @accepts("self", output_channels=int, bottleneck_width=int, stride=[int, tuple], 
        post_trace=False)    
    #@TraceFunction(trace_args=True, trace_rv=True)
    def mobilenet_v2_inverted_linear_bottleneck_block(self, output_channels=32, bottleneck_width=4, stride=1):
        '''
        Append Mobilenet V2 Inverted Linear Bottleneck Block to custom network

        Args:
            output_channels (int): Number of output features for this block
            stride (int): kernel movement stride  
            bottleneck_width (int): Bottleneck dimensions for reducing number of features

        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        
        subnetwork = [];
        branch_1 = [];
        if(bottleneck_width != 1):
            branch_1.append(self.convolution(output_channels=output_channels//bottleneck_width, 
                                            kernel_size=1, stride=1));
        
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        sep_conv = self.separable_convolution_block(input_channels=output_channels//bottleneck_width,
                                                        output_channels=output_channels//bottleneck_width,
                                                        kernel_size=3, stride=stride);
        branch_1.append(sep_conv);   
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=output_channels, kernel_size=1, stride=1));
        branch_1.append(self.batch_normalization());
        
        
        branch_2 = [];
        branch_2.append(self.identity());
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(self.add());
        
        network.append(subnetwork);
        return network;
    #####################################################################################################################################




    #####################################################################################################################################    
    @warning_checks(None, squeeze_channels=None, expand_channels_1x1=None, expand_channels_3x3=None, 
        post_trace=False)
    @error_checks(None, squeeze_channels=["gt", 0], expand_channels_1x1=["gt", 0], expand_channels_3x3=["gt", 0], 
        post_trace=False)
    @accepts("self", squeeze_channels=int, expand_channels_1x1=int, expand_channels_3x3=int, 
        post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def squeezenet_fire_block(self, squeeze_channels=16, expand_channels_1x1=32, expand_channels_3x3=64):
        '''
        Append Squeezenet Fire Block to custom network

        Args:
            squeeze_channels (int): Number of output features for this block
            expand_channels_1x1 (int): Number of convolution_1x1 features for this block
            expand_channels_3x3 (int): Number of convolution_3x3 features for this block
            bottleneck_width (int): Bottleneck dimensions for reducing number of features

        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        
        network.append(self.convolution(output_channels=squeeze_channels, kernel_size=1, stride=1));
        network.append(self.relu());
        
        subnetwork = [];
        branch_1 = [];    
        branch_2 = [];
        
        branch_1.append(self.convolution(output_channels=expand_channels_1x1, kernel_size=1, stride=1));
        branch_1.append(self.relu());
        
        branch_2.append(self.convolution(output_channels=expand_channels_3x3, kernel_size=3, stride=1));
        branch_2.append(self.relu());
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(self.concatenate());
        
        network.append(subnetwork);
        return network;
    #####################################################################################################################################





    #####################################################################################################################################
    @warning_checks(None, bottleneck_size=None, growth_rate=None, dropout=None, 
        post_trace=False)
    @error_checks(None, bottleneck_size=["gt", 0], growth_rate=None, dropout=["gte", 0, "lte", 1], 
        post_trace=False)
    @accepts("self", bottleneck_size=int, growth_rate=int, dropout=[int, float], 
        post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def densenet_block(self, bottleneck_size=4, growth_rate=16, dropout=0.2):
        '''
        Append Densenet Block to custom network

        Args:
            bottleneck_size (int): Bottleneck dimensions for reducing number of features
            growth_rate (int): Expansion rate for convolution layers for this block
            dropout (float): Prbability for dropout layer post convolution

        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        
        subnetwork = [];
        branch_1 = [];
        branch_2 = [];
        
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=bottleneck_size*growth_rate, kernel_size=1, stride=1));
        branch_1.append(self.batch_normalization());
        branch_1.append(self.relu());
        branch_1.append(self.convolution(output_channels=growth_rate, kernel_size=3, stride=1));
        branch_1.append(self.dropout(drop_probability=dropout));
        
        branch_2.append(self.identity());
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(self.concatenate());
        
        network.append(subnetwork);
        
        return network;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, input_channels=None, output_channels=None, kernel_size=None, stride=None, 
        padding=None, post_trace=False)
    @error_checks(None, input_channels=["gt", 0], output_channels=["gt", 0], kernel_size=None, stride=None, 
        padding=None, post_trace=False)
    @accepts("self", input_channels=int, output_channels=int, kernel_size=[int, tuple], stride=[None, int, tuple], 
        padding=[None, int, tuple], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def conv_bn_relu_block(self, output_channels=64, kernel_size=1, stride=1, padding=None):
        '''
        Append Conv->batch_norm->relu Block to custom network

        Args:
            output_channels (int): Number of output features for this block
            kernel_size (int): Kernel matrix shape for all layers in this block
            stride (int): kernel movement stride  
            padding (int, tuple): external zero padding on input
            
        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        if(padding):
            network.append(self.convolution(output_channels=output_channels, 
                                           kernel_size=kernel_size, 
                                           stride=stride,
                                           padding=padding));
        else:
            network.append(self.convolution(output_channels=output_channels, 
                                           kernel_size=kernel_size, 
                                           stride=stride));
        network.append(self.batch_normalization());
        network.append(self.relu());
        
        return network;
    #####################################################################################################################################





    #####################################################################################################################################
    @warning_checks(None, pooling_branch_channels=None, pool_type=None, post_trace=False)
    @error_checks(None, pooling_branch_channels=["gt", 0], pool_type=None, post_trace=False)
    @accepts("self", pooling_branch_channels=int, pool_type=str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def inception_a_block(self, pooling_branch_channels=32, pool_type="avg"):
        '''
        Append Inception-A Block to custom network

        Args:
            pooling_branch_channels (int): Number of features for conv layers in pooling branch
            pool_type (str): Either of these types
                                - "avg" - Average pooling
                                - "max" - Max pooling
            
        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        
        subnetwork = [];
        branch_1 = [];
        branch_2 = [];
        branch_3 = [];
        branch_4 = [];
        
        branch_1.append(self.conv_bn_relu_block(output_channels=64, kernel_size=1))
        
        branch_2.append(self.conv_bn_relu_block(output_channels=48, kernel_size=1));
        branch_2.append(self.conv_bn_relu_block(output_channels=64, kernel_size=5));
           
        branch_3.append(self.conv_bn_relu_block(output_channels=64, kernel_size=1));
        branch_3.append(self.conv_bn_relu_block(output_channels=96, kernel_size=3));
        branch_3.append(self.conv_bn_relu_block(output_channels=96, kernel_size=3));
        
        if(pool_type=="avg"):
            branch_4.append(self.average_pooling(kernel_size=3, stride=1));
        else:
            branch_4.append(self.max_pooling(kernel_size=3, stride=1));
        branch_4.append(self.conv_bn_relu_block(output_channels=pooling_branch_channels, kernel_size=1));
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(branch_3);
        subnetwork.append(branch_4);
        subnetwork.append(self.concatenate());
        
        network.append(subnetwork);
        
        return network;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, pool_type=None, post_trace=False)
    @error_checks(None, pool_type=None, post_trace=False)
    @accepts("self", pool_type=str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def inception_b_block(self, pool_type="avg"):
        '''
        Append Inception-B Block to custom network

        Args:
            pool_type (str): Either of these types
                                - "avg" - Average pooling
                                - "max" - Max pooling
            
        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        
        subnetwork = [];
        branch_1 = [];
        branch_2 = [];
        branch_3 = [];
        branch_4 = [];
        
        branch_1.append(self.conv_bn_relu_block(output_channels=384, kernel_size=3))
              
        branch_2.append(self.conv_bn_relu_block(output_channels=64, kernel_size=1));
        branch_2.append(self.conv_bn_relu_block(output_channels=96, kernel_size=3));
        branch_2.append(self.conv_bn_relu_block(output_channels=96, kernel_size=3));
        
        if(pool_type=="avg"):
            branch_3.append(self.average_pooling(kernel_size=3, stride=1));
        else:
            branch_3.append(self.max_pooling(kernel_size=3, stride=1));
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(branch_3);
        subnetwork.append(self.concatenate());
        
        network.append(subnetwork);

        return network;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, channels_7x7=None, pool_type=None, post_trace=False)
    @error_checks(None, channels_7x7=["gt", 0], pool_type=None, post_trace=False)
    @accepts("self", channels_7x7=int, pool_type=str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def inception_c_block(self, channels_7x7=3, pool_type="avg"):
        '''
        Append Inception-C Block to custom network

        Args:
            channels_7x7 (int): Number of features for conv layers in channels_7x7 branch
            pool_type (str): Either of these types
                                - "avg" - Average pooling
                                - "max" - Max pooling
            
        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        
        subnetwork = [];
        branch_1 = [];
        branch_2 = [];
        branch_3 = [];
        branch_4 = [];
        
        branch_1.append(self.conv_bn_relu_block(output_channels=192, kernel_size=1))
        
        
        branch_2.append(self.conv_bn_relu_block(output_channels=channels_7x7, kernel_size=1));
        branch_2.append(self.conv_bn_relu_block(output_channels=channels_7x7, kernel_size=(1, 7)));
        branch_2.append(self.conv_bn_relu_block(output_channels=192, kernel_size=(7, 1)));
           
            
        branch_3.append(self.conv_bn_relu_block(output_channels=channels_7x7, kernel_size=1));
        branch_3.append(self.conv_bn_relu_block(output_channels=channels_7x7, kernel_size=(1, 7)));
        branch_3.append(self.conv_bn_relu_block(output_channels=channels_7x7, kernel_size=(7, 1)));
        branch_3.append(self.conv_bn_relu_block(output_channels=channels_7x7, kernel_size=(1, 7)));
        branch_3.append(self.conv_bn_relu_block(output_channels=192, kernel_size=(7, 1)));
        
        if(pool_type=="avg"):
            branch_4.append(self.average_pooling(kernel_size=3, stride=1));
        else:
            branch_4.append(self.max_pooling(kernel_size=3, stride=1));
        branch_4.append(self.conv_bn_relu_block(output_channels=192, kernel_size=1));
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(branch_3);
        subnetwork.append(branch_4);
        subnetwork.append(self.concatenate());
        
        network.append(subnetwork);
        
        return network;
    #####################################################################################################################################



    #####################################################################################################################################
    @warning_checks(None, pool_type=None, post_trace=False)
    @error_checks(None, pool_type=None, post_trace=False)
    @accepts("self", pool_type=str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def inception_d_block(self, pool_type="avg"):
        '''
        Append Inception-D Block to custom network

        Args:
            pool_type (str): Either of these types
                                - "avg" - Average pooling
                                - "max" - Max pooling
            
        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        
        subnetwork = [];
        branch_1 = [];
        branch_2 = [];
        branch_3 = [];
        branch_4 = [];
        
        branch_1.append(self.conv_bn_relu_block(output_channels=192, kernel_size=1))
        branch_1.append(self.conv_bn_relu_block(output_channels=320, kernel_size=3, stride=2))
        
        
        branch_2.append(self.conv_bn_relu_block(output_channels=192, kernel_size=1));
        branch_2.append(self.conv_bn_relu_block(output_channels=192, kernel_size=(1, 7)));
        branch_2.append(self.conv_bn_relu_block(output_channels=192, kernel_size=(7, 1)));
        branch_2.append(self.conv_bn_relu_block(output_channels=192, kernel_size=3, stride=2));

        
        if(pool_type=="avg"):
            branch_3.append(self.average_pooling(kernel_size=3, stride=2));
        else:
            branch_3.append(self.max_pooling(kernel_size=3, stride=2));
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(branch_3);
        subnetwork.append(self.concatenate());
        
        network.append(subnetwork);
        
        return network;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, post_trace=False)
    @error_checks(None, post_trace=False)
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def subbranch_block(self):    
        '''
        Append sub-branch Block to custom network

        Args:
            None
            
        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        ''' 
        subnetwork = [];
        branch_1 = [];
        branch_2 = [];
        branch_1.append(self.conv_bn_relu_block(output_channels=384, kernel_size=(1, 3)));
        branch_2.append(self.conv_bn_relu_block(output_channels=384, kernel_size=(3, 1)));
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(self.concatenate());
        return subnetwork;
    #####################################################################################################################################




    #####################################################################################################################################
    @warning_checks(None, pool_type=None, post_trace=False)
    @error_checks(None, pool_type=None, post_trace=False)
    @accepts("self", pool_type=str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def inception_e_block(self, pool_type="avg"):
        '''
        Append Inception-C Block to custom network

        Args:
            pool_type (str): Either of these types
                                - "avg" - Average pooling
                                - "max" - Max pooling
            
        Returns:
            list: Containing all the layer dictionaries arranged as per function arguments
        '''
        network = [];
        
        subnetwork = [];
        branch_1 = [];
        branch_2 = [];
        branch_3 = [];
        branch_4 = [];
        
        branch_1.append(self.conv_bn_relu_block(output_channels=320, kernel_size=1))
        
        branch_2.append(self.conv_bn_relu_block(output_channels=384, kernel_size=1));
        branch_2.append(self.subbranch_block());
        
        
        branch_3.append(self.conv_bn_relu_block(output_channels=448, kernel_size=1));
        branch_3.append(self.conv_bn_relu_block(output_channels=384, kernel_size=3));
        branch_3.append(self.subbranch_block());
        

        
        if(pool_type=="avg"):
            branch_4.append(self.average_pooling(kernel_size=3, stride=1));
        else:
            branch_4.append(self.max_pooling(kernel_size=3, stride=1));
        branch_4.append(self.conv_bn_relu_block(output_channels=192, kernel_size=1));
        
        subnetwork.append(branch_1);
        subnetwork.append(branch_2);
        subnetwork.append(branch_3);
        subnetwork.append(branch_4);
        subnetwork.append(self.concatenate());
        
        network.append(subnetwork);
        
        return network;
    #####################################################################################################################################