from monk.tf_keras_1.finetune.imports import *
from monk.system.imports import *

from monk.tf_keras_1.finetune.level_8_layers_main import prototype_layers


class prototype_transforms(prototype_layers):
    '''
    Main class for all transforms in expert mode

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    @accepts("self", verbose=int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
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
        train=None, val=None, test=None, post_trace=False)
    @accepts("self", brightness=[int, float], contrast=[int, float], saturation=[int, float], hue=[int, float], 
        train=bool, val=bool, test=bool, post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
    def apply_color_jitter(self, brightness=0, contrast=0, saturation=0, hue=0, train=False, val=False, test=False):
        '''
        Apply Color jittering transformations

        Args:
            brightness (float): Levels to jitter brightness.
                                        0 - min
                                        1 - max
            contrast (float): Levels to jitter contrast.
                                        0 - min
                                        1 - max
            saturation (float): Levels to jitter saturation.
                                        0 - min
                                        1 - max
            hue (float): Levels to jitter hue.
                                        0 - min
                                        1 - max
            train (bool): If True, transform applied to training data
            val (bool): If True, transform applied to validation data
            test (bool): If True, transform applied to testing/inferencing data

        Returns:
            None
        '''
        self.system_dict = transform_color_jitter(self.system_dict, brightness, contrast, saturation, hue, train, val, test);
    ###############################################################################################################################################




    ###############################################################################################################################################
    @warning_checks(None, ["lt", 30], translate=["lt", 1.0], scale=["lt", 3.0], shear=None, 
        train=None, val=None, test=None, post_trace=False)
    @error_checks(None, ["gte", 0.0], translate=["gte", 0.0], scale=["gte", 0.0], shear=["gte", 0.0], 
        train=None, val=None, test=None, post_trace=False)
    @accepts("self", [list, float, int], translate=[tuple, type(None)], scale=[tuple, type(None)], 
        shear=[list, float, int, tuple, type(None)], train=bool, val=bool, test=bool, post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_affine(self, degrees, translate=None, scale=None, shear=None, train=False, val=False, test=False):
        '''
        Apply random affine transformations

        Args:
            degrees (float): Max Rotation range limit for transforms
            scale (float, list): Range for randomly scaling 
            shear (float, list): Range for randomly applying sheer changes
            train (bool): If True, transform applied to training data
            val (bool): If True, transform applied to validation data
            test (bool): If True, transform applied to testing/inferencing data

        Returns:
            None
        '''
        self.system_dict = transform_random_affine(self.system_dict, degrees, translate, scale, shear, train, val, test);
    ###############################################################################################################################################



    ###############################################################################################################################################
    @error_checks(None, probability=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=False)
    @accepts("self", probability=float, train=bool, val=bool, test=bool, post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_horizontal_flip(self, probability=0.5, train=False, val=False, test=False):
        '''
        Apply random horizontal flip transformations

        Args:
            probability (float): Probability of flipping the input image
            train (bool): If True, transform applied to training data
            val (bool): If True, transform applied to validation data
            test (bool): If True, transform applied to testing/inferencing data

        Returns:
            None
        '''
        self.system_dict = transform_random_horizontal_flip(self.system_dict, probability, train, val, test);
    ###############################################################################################################################################



    ###############################################################################################################################################
    @error_checks(None, probability=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=False)
    @accepts("self", probability=float, train=bool, val=bool, test=bool, post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_vertical_flip(self, probability=0.5, train=False, val=False, test=False):
        '''
        Apply random vertical flip transformations

        Args:
            probability (float): Probability of flipping the input image
            train (bool): If True, transform applied to training data
            val (bool): If True, transform applied to validation data
            test (bool): If True, transform applied to testing/inferencing data

        Returns:
            None
        '''
        self.system_dict = transform_random_vertical_flip(self.system_dict, probability, train, val, test);
    ###############################################################################################################################################



    ###############################################################################################################################################
    @warning_checks(None, ["lt", 30], train=None, val=None, test=None, post_trace=False)
    @error_checks(None, ["gte", 0.0], train=None, val=None, test=None, post_trace=False)
    @accepts("self", [float, int, list], train=bool, val=bool, test=bool, post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
    def apply_random_rotation(self, degrees, train=False, val=False, test=False):
        '''
        Apply random rotation transformations

        Args:
            degrees (float): Max Rotation range limit for transforms
            train (bool): If True, transform applied to training data
            val (bool): If True, transform applied to validation data
            test (bool): If True, transform applied to testing/inferencing data

        Returns:
            None
        '''
        self.system_dict = transform_random_rotation(self.system_dict, degrees, train, val, test);
    ###############################################################################################################################################




    ###############################################################################################################################################
    @error_checks(None, mean=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=False)
    @accepts("self", mean=[list, float], train=bool, val=bool, test=bool, post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
    def apply_mean_subtraction(self, mean=[123, 116, 103], train=False, val=False, test=False):
        '''
        Apply mean subtraction

        Args:
            mean (float, list): Mean value for subtraction
            train (bool): If True, transform applied to training data
            val (bool): If True, transform applied to validation data
            test (bool): If True, transform applied to testing/inferencing data

        Returns:
            None
        '''
        self.system_dict = transform_mean_subtraction(self.system_dict, mean, train, val, test);
    ###############################################################################################################################################





    ###############################################################################################################################################
    @error_checks(None, mean=["gt", 0, "lt", 1], std=["gt", 0, "lt", 1], train=None, val=None, test=None, post_trace=False)
    @accepts("self", mean=[list, float], std=[list, float], train=bool, val=bool, test=bool, post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
    def apply_normalize(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=False, val=False, test=False):
        '''
        Apply mean subtraction and standard normalization

        Args:
            mean (float, list): Mean value for subtraction
            std (float, list): Normalization factor
            train (bool): If True, transform applied to training data
            val (bool): If True, transform applied to validation data
            test (bool): If True, transform applied to testing/inferencing data

        Returns:
            None
        '''
        self.system_dict = transform_normalize(self.system_dict, mean, std, train, val, test);
    ###############################################################################################################################################
