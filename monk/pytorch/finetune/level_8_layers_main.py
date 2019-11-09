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