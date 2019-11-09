from pytorch.finetune.imports import *
from system.imports import *

from pytorch.finetune.level_8_layers_main import prototype_layers


class prototype_transforms(prototype_layers):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    ###############################################################################################################################################
    @warning_checks(None, ["gte", 32, "lte", 1024], train=None, val=None, test=None, post_trace=True)
    @error_checks(None, ["gt", 0], train=None, val=None, test=None, post_trace=True)
    @accepts("self", int, train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_center_crop(self, input_size, train=False, val=False, test=False):
        self.system_dict = transform_center_crop(self.system_dict, input_size, train, val, test);
    ###############################################################################################################################################



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
    @warning_checks(None, ["gte", 32, "lte", 1024], train=None, val=None, test=None, post_trace=True)
    @error_checks(None, ["gt", 0], train=None, val=None, test=None, post_trace=True)
    @accepts("self", int, train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_crop(self, input_size, train=False, val=False, test=False):    
        self.system_dict = transform_random_crop(self.system_dict, input_size, train, val, test);
    ###############################################################################################################################################



    ###############################################################################################################################################
    @error_checks(None, probability=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=True)
    @accepts("self", probability=float, train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_horizontal_flip(self, probability=0.5, train=False, val=False, test=False):
        self.system_dict = transform_random_horizontal_flip(self.system_dict, probability, train, val, test);
    ###############################################################################################################################################




    ###############################################################################################################################################
    @error_checks(None, distortion_scale=["gt", 0, "lt", 1], probability=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=True)
    @accepts("self", distortion_scale=float, probability=float, train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_perspective(self, distortion_scale=0.5, probability=0.5, train=False, val=False, test=False):
        self.system_dict = transform_random_perspective(self.system_dict, distortion_scale, probability, train, val, test);
    ###############################################################################################################################################





    ###############################################################################################################################################
    @warning_checks(None, ["gte", 32, "lte", 1024], scale=None, ratio=["lt", 2.5], train=None, val=None, test=None, post_trace=True)
    @error_checks(None, ["gt", 0], scale=["gt", 0, "lt", 1], ratio=["gt", 0], train=None, val=None, test=None, post_trace=True)
    @accepts("self", int, scale=[tuple, float], ratio=[tuple, float], train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_resized_crop(self, input_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), train=False, val=False, test=False):
        self.system_dict = transform_random_resized_crop(self.system_dict, input_size, scale, ratio, train, val, test);
    ###############################################################################################################################################




    ###############################################################################################################################################
    @error_checks(None, num_output_channels=["in", [1, 3]], train=None, val=None, test=None, post_trace=True)
    @accepts("self", int, train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_grayscale(self, num_output_channels=3, train=False, val=False, test=False):
        self.system_dict = transform_grayscale(self.system_dict, num_output_channels, train, val, test);
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
    @error_checks(None, probability=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=True)
    @accepts("self", probability=float, train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_vertical_flip(self, probability=0.5, train=False, val=False, test=False):
        self.system_dict = transform_random_vertical_flip(self.system_dict, probability, train, val, test);
    ###############################################################################################################################################



    ###############################################################################################################################################
    @warning_checks(None, ["gte", 32, "lte", 1024], train=None, val=None, test=None, post_trace=True)
    @error_checks(None, ["gt", 0], train=None, val=None, test=None, post_trace=True)
    @accepts("self", int, train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_resize(self, input_size, train=False, val=False, test=False):
        self.system_dict = transform_resize_gluon(self.system_dict, input_size, train, val, test);
    ###############################################################################################################################################


    ###############################################################################################################################################
    @error_checks(None, mean=["gt", 0, "lt", 1], std=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=True)
    @accepts("self", mean=[list, float], std=[list, float], train=bool, val=bool, test=bool, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def apply_normalize(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=False, val=False, test=False):
        self.system_dict = transform_normalize(self.system_dict, mean, std, train, val, test);
    ###############################################################################################################################################



