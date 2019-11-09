from tf_keras.finetune.imports import *
from system.imports import *

from tf_keras.finetune.level_8_layers_main import prototype_layers


class prototype_transforms(prototype_layers):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);
        tmp = {};
        tmp["featurewise_center"] = False;
        tmp["featurewise_std_normalization"] = False;
        tmp["rotation_range"] = 0;
        tmp["width_shift_range"] = 0;
        tmp["height_shift_range"] = 0;
        tmp["shear_range"] = 0;
        tmp["zoom_range"] = 0;
        tmp["brightness_range"] = None;
        tmp["horizontal_flip"] = False;
        tmp["vertical_flip"] = False;
        tmp["mean"] = False;
        tmp["std"] = False;

        self.system_dict["local"]["transforms_train"] = tmp;
        self.system_dict["local"]["transforms_val"] = tmp;
        self.system_dict["local"]["transforms_test"] = tmp;





    ###############################################################################################################################################
    @error_checks(None, brightness=["gte", 0.0], contrast=["gte", 0.0], saturation=["gte", 0.0], hue=["gte", 0.0], 
        train=None, val=None, test=None, post_trace=True)
    @accepts("self", brightness=[int, float], contrast=[int, float], saturation=[int, float], hue=[int, float], 
        train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_color_jitter(self, brightness=0, contrast=0, saturation=0, hue=0, train=False, val=False, test=False):
        self.system_dict = transform_color_jitter(self.system_dict, brightness, contrast, saturation, hue, train, val, test);
    ###############################################################################################################################################




    ###############################################################################################################################################
    @warning_checks(None, ["lt", 30], translate=["lt", 1.0], scale=["lt", 3.0], shear=None, 
        train=None, val=None, test=None, post_trace=True)
    @error_checks(None, ["gte", 0.0], translate=["gte", 0.0], scale=["gte", 0.0], shear=["gte", 0.0], 
        train=None, val=None, test=None, post_trace=True)
    @accepts("self", [list, float, int], translate=[tuple, type(None)], scale=[tuple, type(None)], 
        shear=[list, float, int, tuple, type(None)], train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_affine(self, degrees, translate=None, scale=None, shear=None, train=False, val=False, test=False):
        self.system_dict = transform_random_affine(self.system_dict, degrees, translate, scale, shear, train, val, test);
    ###############################################################################################################################################



    ###############################################################################################################################################
    @error_checks(None, probability=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=True)
    @accepts("self", probability=float, train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_horizontal_flip(self, probability=0.5, train=False, val=False, test=False):
        self.system_dict = transform_random_horizontal_flip(self.system_dict, probability, train, val, test);
    ###############################################################################################################################################



    ###############################################################################################################################################
    @error_checks(None, probability=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=True)
    @accepts("self", probability=float, train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_vertical_flip(self, probability=0.5, train=False, val=False, test=False):
        self.system_dict = transform_random_vertical_flip(self.system_dict, probability, train, val, test);
    ###############################################################################################################################################



    ###############################################################################################################################################
    @warning_checks(None, ["lt", 30], train=None, val=None, test=None, post_trace=True)
    @error_checks(None, ["gte", 0.0], train=None, val=None, test=None, post_trace=True)
    @accepts("self", [float, int, list], train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_rotation(self, degrees, train=False, val=False, test=False):
        self.system_dict = transform_random_rotation(self.system_dict, degrees, train, val, test);
    ###############################################################################################################################################




    ###############################################################################################################################################
    @error_checks(None, mean=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=True)
    @accepts("self", mean=[list, float], train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_mean_subtraction(self, mean=[0.485, 0.456, 0.406], train=False, val=False, test=False):
        self.system_dict = transform_mean_subtraction(self.system_dict, mean, train, val, test);
    ###############################################################################################################################################





    ###############################################################################################################################################
    @error_checks(None, mean=["gt", 0, "lt", 1], std=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=True)
    @accepts("self", mean=[list, float], std=[list, float], train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_normalize(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=False, val=False, test=False):
        self.system_dict = transform_normalize(self.system_dict, mean, std, train, val, test);
    ###############################################################################################################################################
