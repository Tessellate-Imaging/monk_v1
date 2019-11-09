from tf_keras.finetune.imports import *
from system.imports import *

from tf_keras.finetune.level_11_optimizers_main import prototype_optimizers


class prototype_losses(prototype_optimizers):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    ###############################################################################################################################################
    @accepts("self", weight=[list, type(np.array([1, 2, 3])), float, type(None)], size_average=[list, type(np.array([1, 2, 3])), float, type(None)], 
        ignore_index=int, reduction=str, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def loss_crossentropy(self, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
        self.system_dict = categorical_crossentropy(self.system_dict, weight=weight, size_average=size_average, 
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
    def loss_sparse_crossentropy(self, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
        self.system_dict = sparse_categorical_crossentropy(self.system_dict, weight=weight, size_average=size_average, 
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
    def loss_hinge(self, weight=None, size_average=None, ignore_index=-100, reduction='mean'):
        self.system_dict = categorical_hinge(self.system_dict, weight=weight, size_average=size_average, 
            ignore_index=ignore_index, reduction=reduction);
        
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