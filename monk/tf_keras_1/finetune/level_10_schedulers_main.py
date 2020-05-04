from tf_keras_1.finetune.imports import *
from system.imports import *

from tf_keras_1.finetune.level_9_transforms_main import prototype_transforms


class prototype_schedulers(prototype_transforms):
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
    @warning_checks(None, ["gt", 0.01, "lt", 1], last_epoch=None, post_trace=False)
    @error_checks(None, ["gt", 0], last_epoch=None, post_trace=False)
    @accepts("self", [float, int], last_epoch=int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def lr_exponential_decrease(self, gamma, last_epoch=-1):
        '''
        Set learning rate to decrease exponentially every step

        Args:
            gamma (str): Reduction multiplier for reducing learning rate post every step
            last_epoch (int): Set this epoch to a level post which learning rate will not be decreased

        Returns:
            None
        '''
        self.system_dict = scheduler_exponential(self.system_dict, gamma, last_epoch=last_epoch);

        self.custom_print("Learning rate scheduler");
        self.custom_print("    Name:   {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"]));
        self.custom_print("    Params: {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################






    ###############################################################################################################################################
    @warning_checks(None, mode=None, factor=["gt", 0.01, "lt", 1], patience=["lt", 20], verbose=None, threshold=None,
        threshold_mode=None, cooldown=None, min_lr=None, epsilon=["lt", 0.0001], post_trace=False)
    @error_checks(None, mode=["in", ["min", "max"]], factor=["gt", 0], patience=["gt", 0], verbose=None, threshold=["gte", 0], 
        threshold_mode=["in", ["rel", "abs"]], cooldown=["gte", 0], min_lr=["gte", 0], epsilon=["gte", 0], post_trace=False)
    @accepts("self", mode=str, factor=[float, int], patience=int, verbose=bool, threshold=[float, int], 
        threshold_mode=str, cooldown=int, min_lr=[float, list, int], epsilon=float, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def lr_plateau_decrease(self, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, \
        threshold_mode='rel', cooldown=0, min_lr=0, epsilon=1e-08):
        '''
        Set learning rate to decrease if a metric (loss) stagnates in a plateau

        Args:
            mode (str): Either of 
                        - 'min' : lr will be reduced when the quantity monitored (loss) has stopped decreasing; 
                        - 'max' : lr reduced when the quantity monitored (accuracy) has stopped increasing. 
            factor (float): Reduction multiplier for reducing learning rate post every step
            patience (int): Number of epochs to wait before reducing learning rate
            verbose (bool): If True, all computations and wait times are printed
            threshold (float): Preset fixed to 0.0001
            threshold_mode (str): Preset fixed to 'rel' mode
            cooldown (int): Number of epochs to wait before actually applying the scheduler post the actual designated step
            min_lr (float): Set minimum learning rate, post which it will not be decreased
            epsilon (float): A small value to avoid divison by zero.
            last_epoch (int): Set this epoch to a level post which learning rate will not be decreased

        Returns:
            None
        '''
        self.system_dict = scheduler_plateau(self.system_dict, mode=mode, factor=factor, patience=patience, verbose=verbose,
            threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, epsilon=epsilon);

        self.custom_print("Learning rate scheduler");
        self.custom_print("    Name:   {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"]));
        self.custom_print("    Params: {}".format(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################