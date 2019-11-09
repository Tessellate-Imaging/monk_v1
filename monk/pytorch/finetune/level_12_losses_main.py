from pytorch.finetune.imports import *
from system.imports import *

from pytorch.finetune.level_11_optimizers_main import prototype_optimizers


class prototype_losses(prototype_optimizers):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    ###############################################################################################################################################
    @accepts("self", weight=[list, type(np.array([1, 2, 3])), float, type(None)], size_average=[list, type(np.array([1, 2, 3])), float, type(None)], 
        ignore_index=int, reduction=str, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def loss_softmax_crossentropy(self, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
        self.system_dict = softmax_crossentropy(self.system_dict, weight=weight, size_average=size_average, 
            ignore_index=ignore_index, reduction=reduction);
        
        self.custom_print("Loss");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["loss"]["name"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["loss"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################





    ###############################################################################################################################################
    @accepts("self", weight=[list, type(np.array([1, 2, 3])), float, type(None)], size_average=[list, type(np.array([1, 2, 3])), float, type(None)], 
        ignore_index=int, reduction=str, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def loss_nll(self, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
        self.system_dict = nll(self.system_dict, weight=weight, size_average=size_average, 
            ignore_index=ignore_index, reduction=reduction);
        
        self.custom_print("Loss");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["loss"]["name"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["loss"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################




    ###############################################################################################################################################
    @warning_checks(None, log_input=None, full=None, size_average=None, epsilon=["lt", "0.001"], reduce=None, reduction=None, post_trace=True)
    @error_checks(None, log_input=None, full=None, size_average=None, epsilon=["gte", 0], reduce=None, reduction=None, post_trace=True)
    @accepts("self", log_input=bool, full=bool, size_average=[list, type(np.array([1, 2, 3])), float, type(None)], epsilon=[float, int], 
        reduce=type(None), reduction=str, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def loss_poisson_nll(self, log_input=True, full=False, size_average=None, epsilon=1e-08, reduce=None, reduction='mean'):
        self.system_dict = poisson_nll(self.system_dict, log_input=log_input, full=full, size_average=size_average, 
            epsilon=epsilon, reduce=None, reduction=reduction);
        
        self.custom_print("Loss");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["loss"]["name"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["loss"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################





    ###############################################################################################################################################
    @accepts("self", weight=[list, type(np.array([1, 2, 3])), float, type(None)], size_average=[list, type(np.array([1, 2, 3])), float, type(None)], 
        reduce=type(None), reduction=str, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def loss_binary_crossentropy(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        self.system_dict = binary_crossentropy(self.system_dict, weight=weight, size_average=size_average, 
            reduce=None, reduction=reduction);
        
        self.custom_print("Loss");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["loss"]["name"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["loss"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################





    ###############################################################################################################################################
    @accepts("self", weight=[list, type(np.array([1, 2, 3])), float, type(None)], size_average=[list, type(np.array([1, 2, 3])), float, type(None)], 
        reduce=type(None), reduction=str, pos_weight=[list, type(np.array([1, 2, 3])), float, type(None)], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def loss_binary_crossentropy_with_logits(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        self.system_dict = binary_crossentropy_with_logits(self.system_dict, weight=weight, size_average=size_average, 
            reduce=None, reduction=reduction, pos_weight=pos_weight);
        
        self.custom_print("Loss");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["loss"]["name"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["loss"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################