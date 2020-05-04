from gluon.finetune.imports import *
from system.imports import *

from gluon.finetune.level_9_transforms_main import prototype_transforms


class prototype_schedulers(prototype_transforms):
    '''
    Main class for all learning rate schedulers in expert mode

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    @accepts("self", verbose=int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);

    ###############################################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def lr_fixed(self):
        '''
        Set learning rate fixed

        Args:
            None

        Returns:
            None
        '''
        self.system_dict = scheduler_fixed(self.system_dict);
        
        self.custom_print("Learning rate scheduler");
        self.custom_print("    Name:   {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"]));
        self.custom_print("    Params: {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################
        

    ###############################################################################################################################################
    @warning_checks(None, None, gamma=["gt", 0.01, "lt", 1], last_epoch=None, post_trace=False)
    @error_checks(None, ["gt", 0], gamma=["gt", 0], last_epoch=None, post_trace=False)
    @accepts("self", int, gamma=float, last_epoch=int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def lr_step_decrease(self, step_size, gamma=0.1, last_epoch=-1):
        '''
        Set learning rate to decrease in regular steps

        Args:
            step_size (int): Step interval for decreasing learning rate
            gamma (str): Reduction multiplier for reducing learning rate post every step
            last_epoch (int): Set this epoch to a level post which learning rate will not be decreased

        Returns:
            None
        '''
        self.system_dict = scheduler_step(self.system_dict, step_size, gamma=gamma, last_epoch=last_epoch);
        
        self.custom_print("Learning rate scheduler");
        self.custom_print("    Name:   {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"]));
        self.custom_print("    Params: {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################
        

    ###############################################################################################################################################
    @warning_checks(None, None, gamma=["gt", 0.01, "lt", 1], last_epoch=None, post_trace=False)
    @error_checks(None, ["inc", None], gamma=["gt", 0], last_epoch=None, post_trace=False)
    @accepts("self", [list, int], gamma=float, last_epoch=int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def lr_multistep_decrease(self, milestones, gamma=0.1, last_epoch=-1):
        '''
        Set learning rate to decrease in irregular steps

        Args:
            milestones (list): List of epochs at which learning rate is to be decreased
            gamma (str): Reduction multiplier for reducing learning rate post every step
            last_epoch (int): Dummy variable

        Returns:
            None
        '''
        self.system_dict = scheduler_multistep(self.system_dict, milestones, gamma=gamma, last_epoch=last_epoch);
        
        self.custom_print("Learning rate scheduler");
        self.custom_print("    Name:   {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"]));
        self.custom_print("    Params: {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################