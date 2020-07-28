from monk.pytorch.transforms.imports import *
from monk.system.imports import *



@accepts(dict, int, bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_center_crop(system_dict, input_size, train, val, test, retrieve=False):
    '''
    Apply Center Cropping transformation

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        input_size (int, list): Crop size
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["CenterCrop"] = {};
    tmp["CenterCrop"]["input_size"] = input_size;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.CenterCrop(input_size));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.CenterCrop(input_size));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.CenterCrop(input_size));

    return system_dict;



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
    tmp["ColorJitter"]["contrast"] = contrast;
    tmp["ColorJitter"]["saturation"] = saturation;
    tmp["ColorJitter"]["hue"] = hue;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue));


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
        system_dict["local"]["transforms_train"].append(transforms.RandomAffine(degrees, translate=translate, scale=scale, shear=shear));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.RandomAffine(degrees, translate=translate, scale=scale, shear=shear));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.RandomAffine(degrees, translate=translate, scale=scale, shear=shear));

    return system_dict;


@accepts(dict, int, bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_crop(system_dict, input_size, train, val, test, retrieve=True):
    '''
    Apply Random Cropping transformation

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        input_size (int, list): Crop size
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''  
    tmp = {};
    tmp["RandomCrop"] = {};
    tmp["RandomCrop"]["input_size"] = input_size;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.RandomCrop(input_size));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.RandomCrop(input_size));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.RandomCrop(input_size));

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
        system_dict["local"]["transforms_train"].append(transforms.RandomHorizontalFlip(p=probability));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.RandomHorizontalFlip(p=probability));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.RandomHorizontalFlip(p=probability));

    return system_dict;



@accepts(dict, [float, int], [float, int], bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_perspective(system_dict, distortion_scale, probability, train, val, test, retrieve=False):
    '''
    Apply random perspective transformations

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        distortion_scale (float): Max limit for perspective distortion
        probability (float): Probability of applying transformation
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["RandomPerspective"] = {};
    tmp["RandomPerspective"]["distortion_scale"] = distortion_scale;
    tmp["RandomPerspective"]["p"] = probability;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.RandomPerspective(distortion_scale=distortion_scale, p=probability));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.RandomPerspective(distortion_scale=distortion_scale, p=probability));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.RandomPerspective(distortion_scale=distortion_scale, p=probability));

    return system_dict;


@accepts(dict, int, [tuple, list, float], [tuple, list, float], bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_resized_crop(system_dict, input_size, scale, ratio, train, val, test, retrieve=False):
    '''
    Apply Random Resized Cropping transformation

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        input_size (int, list): Crop size
        scale (float, tuple): scaling ratio limits; for maximum and minimum random scaling
        ratio (float, tuple): aspect ratio limits; for maximum and minmum changes to aspect ratios 
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["RandomResizedCrop"] = {};
    tmp["RandomResizedCrop"]["input_size"] = input_size;
    tmp["RandomResizedCrop"]["scale"] = scale;
    tmp["RandomResizedCrop"]["ratio"] = ratio;


    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.RandomResizedCrop(size=input_size, scale=scale, ratio=ratio));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.RandomResizedCrop(size=input_size, scale=scale, ratio=ratio));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.RandomResizedCrop(size=input_size, scale=scale, ratio=ratio));


    return system_dict;


@accepts(dict, int, bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_grayscale(system_dict, num_output_channels, train, val, test, retrieve=False):
    '''
    Not active
    '''
    tmp = {};
    tmp["Grayscale"] = {};
    tmp["Grayscale"]["num_output_channels"] = num_output_channels;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.Grayscale(num_output_channels=num_output_channels));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.Grayscale(num_output_channels=num_output_channels));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.Grayscale(num_output_channels=num_output_channels));

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
        system_dict["local"]["transforms_train"].append(transforms.RandomRotation(degrees));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.RandomRotation(degrees));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.RandomRotation(degrees));

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
        system_dict["local"]["transforms_train"].append(transforms.RandomVerticalFlip(p=probability));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.RandomVerticalFlip(p=probability));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.RandomVerticalFlip(p=probability));

    return system_dict;



@accepts(dict, int, bool, bool, bool, retrieve=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def transform_resize(system_dict, input_size, train, val, test, retrieve=False):
    '''
    Apply standard resizing

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        input_size (int, list): expected final size
        train (bool): If True, transform applied to training data
        val (bool): If True, transform applied to validation data
        test (bool): If True, transform applied to testing/inferencing data

    Returns:
        dict: updated system dict
    '''
    tmp = {};
    tmp["Resize"] = {};
    tmp["Resize"]["input_size"] = input_size;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.Resize(size=(input_size, input_size)));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.Resize(size=(input_size, input_size)));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.Resize(size=(input_size, input_size)));

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


    if(type(system_dict["dataset"]["params"]["input_size"]) == tuple or type(system_dict["dataset"]["params"]["input_size"]) == list):
        h = system_dict["dataset"]["params"]["input_size"][0];
        w = system_dict["dataset"]["params"]["input_size"][1];
    else:
        h = system_dict["dataset"]["params"]["input_size"];
        w = system_dict["dataset"]["params"]["input_size"];

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.Resize(size=(w, h)));
        system_dict["local"]["transforms_train"].append(transforms.ToTensor())
        system_dict["local"]["transforms_train"].append(transforms.Normalize(mean=mean, std=std));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.Resize(size=(w, h)));
        system_dict["local"]["transforms_val"].append(transforms.ToTensor())
        system_dict["local"]["transforms_val"].append(transforms.Normalize(mean=mean, std=std));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.Resize(size=(w, h)));
        system_dict["local"]["transforms_test"].append(transforms.ToTensor())
        system_dict["local"]["transforms_test"].append(transforms.Normalize(mean=mean, std=std));

    return system_dict;

