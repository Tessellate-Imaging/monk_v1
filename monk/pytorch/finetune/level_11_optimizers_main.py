from pytorch.finetune.imports import *
from system.imports import *

from pytorch.finetune.level_10_schedulers_main import prototype_schedulers


class prototype_optimizers(prototype_schedulers):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], momentum=["lt", 1.5], weight_decay=["lt", 0.01], momentum_dampening_rate=None, 
        clipnorm=None, clipvalue=None, post_trace=True)
    @error_checks(None, ["gt", 0], momentum=["gte", 0], weight_decay=["gte", 0], momentum_dampening_rate=None, 
        clipnorm=None, clipvalue=None, post_trace=True)
    @accepts("self", [int, float], momentum=[int, float], weight_decay=[int, float], momentum_dampening_rate=[int, float], 
        clipnorm=[int, float], clipvalue=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_sgd(self, learning_rate, momentum=0, weight_decay=0, momentum_dampening_rate=0, clipnorm=0.0, clipvalue=0.0):
        self.system_dict = sgd(self.system_dict, learning_rate, 
                momentum=momentum, weight_decay=weight_decay, momentum_dampening_rate=momentum_dampening_rate, clipnorm=clipnorm, clipvalue=clipvalue);
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: clipnorm and clipvalue are active only for keras in current version of Monk");
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: momentum_dampening_rate is active only for pytorch in current version of Monk");
        self.custom_print("");
    ###############################################################################################################################################


    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], momentum=["lt", 1.5], weight_decay=["lt", 0.01], momentum_dampening_rate=None, 
        clipnorm=None, clipvalue=None, post_trace=True)
    @error_checks(None, ["gt", 0], momentum=["gte", 0], weight_decay=["gte", 0], momentum_dampening_rate=None, 
        clipnorm=None, clipvalue=None, post_trace=True)
    @accepts("self", [int, float], momentum=[int, float], weight_decay=[int, float], momentum_dampening_rate=[int, float], 
        clipnorm=[int, float], clipvalue=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_nesterov_sgd(self, learning_rate, momentum=0, weight_decay=0, momentum_dampening_rate=0, clipnorm=0.0, clipvalue=0.0):
        self.system_dict = nesterov_sgd(self.system_dict, learning_rate, 
                momentum=momentum, weight_decay=weight_decay, momentum_dampening_rate=momentum_dampening_rate, clipnorm=clipnorm, clipvalue=clipvalue);
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: clipnorm and clipvalue are active only for keras in current version of Monk");
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: momentum_dampening_rate is active only for pytorch in current version of Monk");
        self.custom_print("");
    ###############################################################################################################################################


    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], decay_rate=["lt", 1], epsilon=["lt", 0.001], weight_decay=["lt", 0.01], 
        clipnorm=None, clipvalue=None, post_trace=None)
    @error_checks(None, ["gt", 0], decay_rate=["gt", 0], epsilon=["gte", 0], weight_decay=["gte", 0], 
        clipnorm=None, clipvalue=None, post_trace=True)
    @accepts("self", [int, float], decay_rate=[int, float], epsilon=[int, float], weight_decay=[int, float], 
        clipnorm=[int, float], clipvalue=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_rmsprop(self, learning_rate, decay_rate=0.99, epsilon=1e-08, weight_decay=0, clipnorm=0.0, clipvalue=0.0):
        self.system_dict = rmsprop(self.system_dict , learning_rate, 
            decay_rate=decay_rate, epsilon=epsilon, weight_decay=weight_decay, clipnorm=clipnorm, clipvalue=clipvalue);
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: clipnorm and clipvalue are active only for keras in current version of Monk");
        self.custom_print("");
    ###############################################################################################################################################


    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], decay_rate=["lt", 1], epsilon=["lt", 0.001], weight_decay=["lt", 0.01], 
        momentum=["lt", 1.5], post_trace=None)
    @error_checks(None, ["gt", 0], decay_rate=["gt", 0], epsilon=["gte", 0], weight_decay=["gte", 0], 
        momentum=["gte", 0], post_trace=True)
    @accepts("self", [int, float], decay_rate=[int, float], epsilon=[int, float], weight_decay=[int, float], 
        momentum=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_momentum_rmsprop(self, learning_rate, decay_rate=0.99, epsilon=1e-08, weight_decay=0, momentum=0.9):
        self.system_dict = momentum_rmsprop(self.system_dict , learning_rate, 
            decay_rate=decay_rate, epsilon=epsilon, weight_decay=weight_decay, momentum=momentum);
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
    ###############################################################################################################################################


    ###############################################################################################################################################
    @warning_checks(None, ["lt, 1"], beta1=["lt", 1], beta2=["lt", 1], epsilon=["lt", 0.001],  weight_decay=["lt", 0.01], amsgrad=None, 
        clipnorm=None, clipvalue=None, post_trace=True)
    @error_checks(None, ["gt", 0], beta1=["gte", 0], beta2=["gte", 0], epssilon=["gte", 0], weight_decay=["gte", 0], amsgrad=None, 
        clipnorm=None, clipvalue=None, post_trace=True)
    @accepts("self", [int, float], beta1=[int, float], beta2=[int, float], epsilon=[int, float], weight_decay=[int, float], amsgrad=bool, 
        clipnorm=[int, float], clipvalue=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_adam(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, weight_decay=0, amsgrad=False, 
        clipnorm=0.0, clipvalue=0.0):
        self.system_dict = adam(self.system_dict, learning_rate,
            beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay, amsgrad=amsgrad, clipnorm=clipnorm, clipvalue=clipvalue);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: clipnorm and clipvalue are active only for keras in current version of Monk");
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: amsgrad is active only for keras and pytorch in current version of Monk");
        self.custom_print("");
    ###############################################################################################################################################


    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], beta1=["lt", 1], beta2=["lt", 1], epsilon=["lt", 0.001], weight_decay=["lt", 0.01], 
        clipnorm=None, clipvalue=None, post_trace=True)
    @error_checks(None, ["gt", 0], beta1=["gte", 0], beta2=["gte", 0], epsilon=["gte", 0], weight_decay=["gte", 0], 
        clipnorm=None, clipvalue=None, post_trace=True)
    @accepts("self", [int, float], beta1=[int, float], beta2=[int, float], epsilon=[int, float], weight_decay=[int, float], 
        clipnorm=[int, float], clipvalue=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_adamax(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, weight_decay=0,
        clipnorm=0.0, clipvalue=0.0):

        self.system_dict = adamax(self.system_dict, learning_rate,
            beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay, clipnorm=clipnorm, clipvalue=clipvalue);  
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: clipnorm and clipvalue are active only for keras in current version of Monk");
        self.custom_print("");
    ###############################################################################################################################################


    ###############################################################################################################################################
    @warning_checks(None, ["lt, 1"], beta1=["lt", 1], beta2=["lt", 1], epsilon=["lt", 0.001],  weight_decay=["lt", 0.01], amsgrad=None, post_trace=True)
    @error_checks(None, ["gt", 0], beta1=["gte", 0], beta2=["gte", 0], epssilon=["gte", 0], weight_decay=["gte", 0], amsgrad=None, post_trace=True)
    @accepts("self", [int, float], beta1=[int, float], beta2=[int, float], epsilon=[int, float], weight_decay=[int, float], amsgrad=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_adamw(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, weight_decay=0, amsgrad=False):
        self.system_dict = adamw(self.system_dict, learning_rate,
            beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay, amsgrad=amsgrad);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
        ConstraintWarning("OptimizerWarning: adamw is active only for pytorch in current version of Monk");
        self.custom_print("");
    ###############################################################################################################################################


    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], rho=["lt", 1], epsilon=["lt", 0.001], weight_decay=["lt", 0.01], 
        clipnorm=None, clipvalue=None, post_trace=True)
    @error_checks(None, ["gt", 0], rho=["gt", 0], epsilon=["gte", 0], weight_decay=["gte", 0], 
        clipnorm=None, clipvalue=None, post_trace=True)
    @accepts("self", [int, float], rho=[int, float], epsilon=[int, float], weight_decay=[int, float], 
        clipnorm=[int, float], clipvalue=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_adadelta(self, learning_rate, rho=0.9, epsilon=1e-06, weight_decay=0, 
        clipnorm=0.0, clipvalue=0.0):
        self.system_dict = adadelta(self.system_dict, learning_rate,
            rho=rho, epsilon=epsilon, weight_decay=weight_decay, clipnorm=clipnorm, clipvalue=clipvalue);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: clipnorm and clipvalue are active only for keras in current version of Monk");
        self.custom_print("");
    ###############################################################################################################################################



    ###############################################################################################################################################
    @warning_checks(None, ["lt", 1], learning_rate_decay=None, weight_decay=["lt", 0.01], epsilon=None, 
        clipnorm=None, clipvalue=None, post_trace=True)
    @error_checks(None, ["gt", 0], learning_rate_decay=None, weight_decay=["gte", 0], epsilon=None, 
        clipnorm=None, clipvalue=None, post_trace=True)
    @accepts("self", [int, float], learning_rate_decay=[int, float], weight_decay=[int, float], 
        epsilon=[int, float], clipnorm=[int, float], clipvalue=[int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def optimizer_adagrad(self, learning_rate, learning_rate_decay=0, weight_decay=0, epsilon=1e-08,
        clipnorm=0.0, clipvalue=0.0):
        self.system_dict = adagrad(self.system_dict, learning_rate,
            learning_rate_decay=learning_rate_decay, weight_decay=weight_decay, epsilon=epsilon, 
            clipnorm=clipnorm, clipvalue=clipvalue);
        
        self.custom_print("Optimizer");
        self.custom_print("    Name:          {}".format(self.system_dict["hyper-parameters"]["optimizer"]["name"]));
        self.custom_print("    Learning rate: {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("    Params:        {}".format(self.system_dict["hyper-parameters"]["optimizer"]["params"]));
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: clipnorm and clipvalue are active only for keras in current version of Monk");
        self.custom_print("");
        ConstraintWarning("ArgumentWarning: learning_rate_decay is active only for pytorch in current version of Monk");
        self.custom_print("");
    ###############################################################################################################################################


