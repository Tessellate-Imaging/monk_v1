from monk.tf_keras_1.transforms.imports import *
from monk.system.imports import *




@accepts(dict, [list, float, int], [list, float, int], [list, float, int], [list, float, int], bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_color_jitter(system_dict, brightness, contrast, saturation, hue, train, val, test, retrieve=False):
    '''
    Apply Color jittering transformations

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
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
        dict: updated system dict
    '''
    tmp = {};
    tmp["ColorJitter"] = {};
    tmp["ColorJitter"]["brightness"] = brightness;

    if(contrast or saturation or hue):
        msg = "Unimplemented - contrast, saturation, hue.\n";
        ConstraintWarning(msg);


    tmp["ColorJitter"]["contrast"] = contrast;
    tmp["ColorJitter"]["saturation"] = saturation;
    tmp["ColorJitter"]["hue"] = hue;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"]["brightness_range"] = [max(0, 1-brightness), 1+brightness];
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"]["brightness_range"] = [max(0, 1-brightness), 1+brightness];
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"]["brightness_range"] = [max(0, 1-brightness), 1+brightness];

    return system_dict;






@accepts(dict, [list, float, int], [tuple, list, type(None)], [tuple, list, type(None)], [list, float, int, tuple, type(None)], 
    bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_affine(system_dict, degrees, translate, scale, shear, train, val, test, retrieve=False):
    '''
    Apply random affine transformations

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        degrees (float): Max Rotation range limit for transforms
        scale (float, list): Range for randomly scaling 
        shear (float, list): Range for randomly applying sheer changes
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["RandomAffine"] = {};
    tmp["RandomAffine"]["degrees"] = degrees;
    tmp["RandomAffine"]["translate"] = translate;
    tmp["RandomAffine"]["scale"] = scale;
    tmp["RandomAffine"]["shear"] = shear;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"]["rotation_range"] = degrees;
        system_dict["local"]["transforms_train"]["width_shift_range"] = translate;
        system_dict["local"]["transforms_train"]["height_shift_range"] = degrees;
        system_dict["local"]["transforms_train"]["zoom_range"] = scale;
        system_dict["local"]["transforms_train"]["shear_range"] = shear;
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"]["rotation_range"] = degrees;
        system_dict["local"]["transforms_val"]["width_shift_range"] = translate;
        system_dict["local"]["transforms_val"]["height_shift_range"] = degrees;
        system_dict["local"]["transforms_val"]["zoom_range"] = scale;
        system_dict["local"]["transforms_val"]["shear_range"] = shear;
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"]["rotation_range"] = degrees;
        system_dict["local"]["transforms_test"]["width_shift_range"] = translate;
        system_dict["local"]["transforms_test"]["height_shift_range"] = degrees;
        system_dict["local"]["transforms_test"]["zoom_range"] = scale;
        system_dict["local"]["transforms_test"]["shear_range"] = shear;

    return system_dict;





@accepts(dict, float, bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_horizontal_flip(system_dict, probability, train, val, test, retrieve=False):
    '''
    Apply random horizontal flip transformations

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        probability (float): Probability of flipping the input image
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["RandomHorizontalFlip"] = {};
    tmp["RandomHorizontalFlip"]["p"] = probability;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"]["horizontal_flip"] = True;
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"]["horizontal_flip"] = True;
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"]["horizontal_flip"] = True;

    return system_dict;





@accepts(dict, float, bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_vertical_flip(system_dict, probability, train, val, test, retrieve=False):
    '''
    Apply random vertical flip transformations

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        probability (float): Probability of flipping the input image
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["RandomVerticalFlip"] = {};
    tmp["RandomVerticalFlip"]["p"] = probability;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"]["horizontal_flip"] = True;
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"]["horizontal_flip"] = True;
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"]["horizontal_flip"] = True;

    return system_dict;






@accepts(dict, [float, int, list], bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_rotation(system_dict, degrees, train, val, test, retrieve=False):
    '''
    Apply random rotation transformations

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        degrees (float): Max Rotation range limit for transforms
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["RandomRotation"] = {};
    tmp["RandomRotation"]["degrees"] = degrees;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"]["rotation_range"] = degrees;
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"]["rotation_range"] = degrees;
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"]["rotation_range"] = degrees;

    return system_dict;





@accepts(dict, [float, list], bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_mean_subtraction(system_dict, mean, train, val, test, retrieve=False):
    '''
    Apply mean subtraction

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        mean (float, list): Mean value for subtraction
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["MeanSubtraction"] = {};
    tmp["MeanSubtraction"]["mean"] = mean;
    system_dict["local"]["mean_subtract"] = True;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"]["mean"] = np.array(mean)*255;
        system_dict["local"]["transforms_train"]["featurewise_center"] = True;
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"]["mean"] = np.array(mean)*255;
        system_dict["local"]["transforms_val"]["featurewise_center"] = True;
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"]["mean"] = np.array(mean)*255;
        system_dict["local"]["transforms_test"]["featurewise_center"] = True;

    return system_dict;







@accepts(dict, [float, list], [float, list], bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_normalize(system_dict, mean, std, train, val, test, retrieve=False):
    '''
    Apply mean subtraction and standard normalization

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        mean (float, list): Mean value for subtraction
        std (float, list): Normalization factor
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["Normalize"] = {};
    tmp["Normalize"]["mean"] = mean;
    tmp["Normalize"]["std"] = std;
    system_dict["local"]["normalize"] = True;
    input_size = system_dict["dataset"]["params"]["input_size"];

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"]["mean"] = np.array(mean)*255;
        system_dict["local"]["transforms_train"]["std"] = np.array(std)*255;
        system_dict["local"]["transforms_train"]["featurewise_center"] = True;
        system_dict["local"]["transforms_train"]["featurewise_std_normalization"] = True;
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"]["mean"] = np.array(mean)*255;
        system_dict["local"]["transforms_val"]["std"] = np.array(std)*255;
        system_dict["local"]["transforms_val"]["featurewise_center"] = True;
        system_dict["local"]["transforms_val"]["featurewise_std_normalization"] = True;
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"]["mean"] = np.array(mean)*255;
        system_dict["local"]["transforms_test"]["std"] = np.array(std)*255;
        system_dict["local"]["transforms_test"]["featurewise_center"] = True;
        system_dict["local"]["transforms_test"]["featurewise_std_normalization"] = True;

    return system_dict;