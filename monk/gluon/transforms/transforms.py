from gluon.transforms.imports import *
from system.imports import *



@accepts(dict, int, [tuple, list, float], [tuple, list, float], bool, bool, bool, retrieve=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_resized_crop(system_dict, input_size, scale, ratio, train, val, test, retrieve=False):
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


@accepts(dict, int, bool, bool, bool, retrieve=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def transform_center_crop(system_dict, input_size, train, val, test, retrieve=False):
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


@accepts(dict, [list, float, int], [list, float, int], [list, float, int], [list, float, int], bool, bool, bool, retrieve=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def transform_color_jitter(system_dict, brightness, contrast, saturation, hue, train, val, test, retrieve=False):
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



@accepts(dict, float, bool, bool, bool, retrieve=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_horizontal_flip(system_dict, probability, train, val, test, retrieve=False):
    tmp = {};
    tmp["RandomHorizontalFlip"] = {};
    tmp["RandomHorizontalFlip"]["p"] = probability;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.RandomFlipLeftRight());
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.RandomFlipLeftRight());
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.RandomFlipLeftRight());


    return system_dict;


@accepts(dict, float, bool, bool, bool, retrieve=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_vertical_flip(system_dict, probability, train, val, test, retrieve=False):
    tmp = {};
    tmp["RandomVerticalFlip"] = {};
    tmp["RandomVerticalFlip"]["p"] = probability;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.RandomFlipTopBottom());
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.RandomFlipTopBottom());
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.RandomFlipTopBottom());

    return system_dict;


@accepts(dict, [float, int], bool, bool, bool, retrieve=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def transform_random_lighting(system_dict, alpha, train, val, test, retrieve=False):
    tmp = {};
    tmp["RandomLighting"] = {};
    tmp["RandomLighting"]["alpha"] = alpha;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.RandomLighting(alpha));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.RandomLighting(alpha));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.RandomLighting(alpha));


    return system_dict;


@accepts(dict, int, bool, bool, bool, retrieve=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def transform_resize(system_dict, input_size, train, val, test, retrieve=False):
    tmp = {};
    tmp["Resize"] = {};
    tmp["Resize"]["input_size"] = input_size;

    if(type(input_size) == tuple or type(input_size) == list):
        h = input_size[0];
        w = input_size[1];
    else:
        h = input_size;
        w = input_size;

    if(train):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["train"].append(tmp);
        system_dict["local"]["transforms_train"].append(transforms.Resize(size=(w, h)));
    if(val):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["val"].append(tmp);
        system_dict["local"]["transforms_val"].append(transforms.Resize(size=(w, h)));
    if(test):
        if(not retrieve):
            system_dict["dataset"]["transforms"]["test"].append(tmp);
        system_dict["local"]["transforms_test"].append(transforms.Resize(size=(w, h)));

    return system_dict;


@accepts(dict, [float, list], [float, list], bool, bool, bool, retrieve=bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def transform_normalize(system_dict, mean, std, train, val, test, retrieve=False):
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