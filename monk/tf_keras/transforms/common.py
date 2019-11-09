from tf_keras.transforms.imports import *
from system.imports import *

from tf_keras.transforms.transforms import transform_color_jitter
from tf_keras.transforms.transforms import transform_random_affine
from tf_keras.transforms.transforms import transform_random_horizontal_flip
from tf_keras.transforms.transforms import transform_random_rotation
from tf_keras.transforms.transforms import transform_random_vertical_flip
from tf_keras.transforms.transforms import transform_mean_subtraction
from tf_keras.transforms.transforms import transform_normalize



@accepts(dict, list, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_transforms(system_dict, set_phases):
    transforms_test = [];
    transforms_train = [];
    transforms_val = [];
    transformations = system_dict["dataset"]["transforms"];
    normalize = False;
    for phase in set_phases:
        tsf = transformations[phase];
        if(phase=="train"):
            train_status = True;
            val_status = False;
            test_status = False;
        elif(phase=="val"):
            train_status = False;
            val_status = True;
            test_status = False;
        else:
            train_status = False;
            val_status = False;
            test_status = True;

        for i in range(len(tsf)):
            name = list(tsf[i].keys())[0]
            input_dict = tsf[i][name];
            train = train_status;
            val = val_status;
            test = test_status;

            if(name == "ColorJitter"):
                system_dict = transform_color_jitter(
                    system_dict, 
                    input_dict["brightness"], input_dict["contrast"], input_dict["saturation"], input_dict["hue"],
                    train, val, test, retrieve=True
                    );
            elif(name == "RandomAffine"):
                system_dict = transform_random_affine(
                    system_dict, 
                    input_dict["degrees"], input_dict["translate"], input_dict["scale"], input_dict["shear"], 
                    train, val, test, retrieve=True
                    );
            elif(name == "RandomHorizontalFlip"):
                system_dict = transform_random_horizontal_flip(
                    system_dict, 
                    input_dict["p"],
                    train, val, test, retrieve=True
                    );
            elif(name == "RandomVerticalFlip"):
                system_dict = transform_random_vertical_flip(
                    system_dict, 
                    input_dict["p"],
                    train, val, test, retrieve=True
                    );
            elif(name == "RandomRotation"):
                system_dict = transform_random_rotation(
                    system_dict, 
                    input_dict["degrees"], 
                    train, val, test, retrieve=True
                    );
            elif(name == "MeanSubtraction"):
                system_dict = transform_mean_subtraction(
                    system_dict,
                    input_dict["mean"], 
                    train, val, test, retrieve=True
                    );
            elif(name == "Normalize"):
                system_dict = transform_normalize(
                    system_dict,
                    input_dict["mean"], input_dict["std"],
                    train, val, test, retrieve=True
                    );


    return system_dict;


