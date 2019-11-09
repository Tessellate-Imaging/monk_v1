from gluon.finetune.imports import *
from system.imports import *

from gluon.finetune.level_9_transforms_main import prototype_transforms


class prototype_schedulers(prototype_transforms):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);

    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def lr_fixed(self):
        self.system_dict = scheduler_fixed(self.system_dict);
        
        self.custom_print("Learning rate scheduler");
        self.custom_print("    Name:   {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"]));
        self.custom_print("    Params: {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################
        

    ###############################################################################################################################################
    @warning_checks(None, None, gamma=["gt", 0.01, "lt", 1], last_epoch=None, post_trace=True)
    @error_checks(None, ["gt", 0], gamma=["gt", 0], last_epoch=None, post_trace=True)
    @accepts("self", int, gamma=float, last_epoch=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def lr_step_decrease(self, step_size, gamma=0.1, last_epoch=-1):
        self.system_dict = scheduler_step(self.system_dict, step_size, gamma=gamma, last_epoch=last_epoch);
        
        self.custom_print("Learning rate scheduler");
        self.custom_print("    Name:   {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"]));
        self.custom_print("    Params: {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################
        

    ###############################################################################################################################################
    @warning_checks(None, None, gamma=["gt", 0.01, "lt", 1], last_epoch=None, post_trace=True)
    @error_checks(None, ["inc", None], gamma=["gt", 0], last_epoch=None, post_trace=True)
    @accepts("self", [list, int], gamma=float, last_epoch=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def lr_multistep_decrease(self, milestones, gamma=0.1, last_epoch=-1):
        self.system_dict = scheduler_multistep(self.system_dict, milestones, gamma=gamma, last_epoch=last_epoch);
        
        self.custom_print("Learning rate scheduler");
        self.custom_print("    Name:   {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"]));
        self.custom_print("    Params: {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################