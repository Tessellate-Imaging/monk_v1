from pytorch.finetune.imports import *
from system.imports import *

from pytorch.finetune.level_10_schedulers_main import prototype_schedulers


class prototype_optimizers(prototype_schedulers):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], rho=["lt", 1], epsilon=["lt", 0.001], weight_decay=["lt", 0.01], post_trace=True)
    @error_checks(None, ["gt", 0], rho=["gt", 0], epsilon=["gte", 0], weight_decay=["gte", 0], post_trace=True)
    @accepts("self", [int, float], rho=[int, float], epsilon=[int, float], weight_decay=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_adadelta(self, learning_rate, rho=0.9, epsilon=1e-06, weight_decay=0):
        self.system_dict = adadelta(self.system_dict, learning_rate,
            rho=rho, epsilon=epsilon, weight_decay=weight_decay);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################




    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], learning_rate_decay=None, weight_decay=["lt", 0.01], initial_accumulator_value=None, post_trace=True)
    @error_checks(None, ["gt", 0], learning_rate_decay=None, weight_decay=["gte", 0], initial_accumulator_value=None, post_trace=True)
    @accepts("self", [int, float], learning_rate_decay=[int, float], weight_decay=[int, float], 
        initial_accumulator_value=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_adagrad(self, learning_rate, learning_rate_decay=0, weight_decay=0, initial_accumulator_value=0):
        self.system_dict = adagrad(self.system_dict, learning_rate,
            learning_rate_decay=learning_rate_decay, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################





    ###############################################################################################################################################
    @warning_checks(None, ["lt, 1"], betas=["lt", 1], epsilon=["lt", 0.001],  weight_decay=["lt", 0.01], amsgrad=None, post_trace=True)
    @error_checks(None, ["gt", 0], betas=["gte", 0], epssilon=["gte", 0], weight_decay=["gte", 0], amsgrad=None, post_trace=True)
    @accepts("self", [int, float], betas=tuple, epsilon=[int, float], weight_decay=[int, float], amsgrad=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_adam(self, learning_rate, betas=(0.9, 0.999), epsilon=1e-08, weight_decay=0, amsgrad=False):
        self.system_dict = adam(self.system_dict, learning_rate,
            betas=betas, epsilon=epsilon, weight_decay=weight_decay, amsgrad=amsgrad);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################





    ###############################################################################################################################################
    @warning_checks(None, ["lt, 1"], betas=["lt", 1], epsilon=["lt", 0.001],  weight_decay=["lt", 0.01], amsgrad=None, post_trace=True)
    @error_checks(None, ["gt", 0], betas=["gte", 0], epssilon=["gte", 0], weight_decay=["gte", 0], amsgrad=None, post_trace=True)
    @accepts("self", [int, float], betas=tuple, epsilon=[int, float], weight_decay=[int, float], amsgrad=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_adamw(self, learning_rate, betas=(0.9, 0.999), epsilon=1e-08, weight_decay=0, amsgrad=False):
        self.system_dict = adamw(self.system_dict, learning_rate,
            betas=betas, epsilon=epsilon, weight_decay=weight_decay, amsgrad=amsgrad);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################





    ###############################################################################################################################################
    @warning_checks(None, ["lt, 1"], betas=["lt", 1], epsilon=["lt", 0.001], post_trace=True)
    @error_checks(None, ["gt", 0], betas=["gte", 0], epssilon=["gte", 0], post_trace=True)
    @accepts("self", [int, float], betas=tuple, epsilon=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_sparseadam(self, learning_rate, betas=(0.9, 0.999), epsilon=1e-08):
        self.system_dict = sparseadam(self.system_dict, learning_rate,
            betas=betas, epsilon=epsilon);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################





    ###############################################################################################################################################
    @warning_checks(None, ["lt, 1"], betas=["lt", 1], epsilon=["lt", 0.001],  weight_decay=["lt", 0.01], post_trace=True)
    @error_checks(None, ["gt", 0], betas=["gte", 0], epssilon=["gte", 0], weight_decay=["gte", 0], post_trace=True)
    @accepts("self", [int, float], betas=tuple, epsilon=[int, float], weight_decay=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_adamax(self, learning_rate, betas=(0.9, 0.999), epsilon=1e-08, weight_decay=0):
        self.system_dict = adamax(self.system_dict, learning_rate,
            betas=betas, epsilon=epsilon, weight_decay=weight_decay);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################





    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], lambd=None, alpha=None, t0=None, weight_decay=["lt", 0.01], post_trace=True)
    @error_checks(None, ["gt", 0], lambd=["gt", 0], alpha=["gt", 0], t0=None, weight_decay=["gte", 0], post_trace=True)
    @accepts("self", [int, float], lambd=[float, int], alpha=[float, int], t0=[float, int], weight_decay=[int, float])
    @TraceFunction(trace_args=False, trace_rv=False)
    def optimizer_asgd(self, learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0):
        self.system_dict = asgd(self.system_dict, learning_rate,
            lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)

        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################




    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], alpha=["lt", 1], epsilon=["lt", 0.001], weight_decay=["lt", 0.01], momentum=["lt", 1], 
        centered=None, post_trace=None)
    @error_checks(None, ["gt", 0], alpha=["gt", 0], epsilon=["gte", 0], weight_decay=["gte", 0], momentum=["gt", 0], 
        centered=None, post_trace=True)
    @accepts("self", [int, float], alpha=[int, float], epsilon=[int, float], weight_decay=[int, float], momentum=[int, float], 
        centered=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_rmsprop(self, learning_rate, alpha=0.99, epsilon=1e-08, weight_decay=0, momentum=0, centered=False):
        if(momentum > 0 and not centered):
            msg = "Momentum not applied as centered flag = Flag\n.";
            msg += "To activate momentum set centered = True.";
            ConstraintWarning(msg);

        self.system_dict = rmsprop(self.system_dict , learning_rate, 
            alpha=alpha, epsilon=epsilon, weight_decay=weight_decay, momentum=momentum, centered=centered);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################






    ###############################################################################################################################################
    @error_checks(None, ["gt", 0], etas=["gt", 0], step_sizes=["gt", 0], post_trace=True)
    @accepts("self", [int, float], etas=tuple, step_sizes=tuple, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_rprop(self, learning_rate, etas=(0.5, 1.2), step_sizes=(1e-06, 50)):
        self.system_dict = rprop(self.system_dict, learning_rate,
            etas=etas, step_sizes=step_sizes);

        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################





    ###############################################################################################################################################                                                                                                                                 
    @warning_checks(None, ["lt", 1], momentum=["lt", 1.5], dampening=None, weight_decay=["lt", 0.01], nesterov=None, post_trace=True)
    @error_checks(None, ["gt", 0], momentum=["gte", 0], dampening=None, weight_decay=["gte", 0], nesterov=None, post_trace=True)
    @accepts("self", [int, float], momentum=[int, float], dampening=[int, float], weight_decay=[int, float], nesterov=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_sgd(self, learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        self.system_dict = sgd(self.system_dict, learning_rate, 
                momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov);
            
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################
