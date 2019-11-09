from tf_keras.transforms.imports import *
from system.imports import *



@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_transform_estimate(system_dict):

    if(system_dict["local"]["transforms_train"]["featurewise_center"]):
        rescale_train = 1/256;
    else:
        rescale_train = 0;
    if(system_dict["local"]["transforms_val"]["featurewise_center"]):
        rescale_val = 1/256;
    else:
        rescale_val = 0;


    system_dict["local"]["data_generators"]["estimate"] = keras.preprocessing.image.ImageDataGenerator(
                            featurewise_center=system_dict["local"]["transforms_train"]["featurewise_center"],
                            featurewise_std_normalization=system_dict["local"]["transforms_train"]["featurewise_std_normalization"],
                            rotation_range=system_dict["local"]["transforms_train"]["rotation_range"],
                            width_shift_range=system_dict["local"]["transforms_train"]["width_shift_range"],
                            height_shift_range=system_dict["local"]["transforms_train"]["height_shift_range"],
                            shear_range=system_dict["local"]["transforms_train"]["shear_range"],
                            zoom_range=system_dict["local"]["transforms_train"]["zoom_range"],
                            brightness_range=system_dict["local"]["transforms_train"]["brightness_range"],
                            horizontal_flip=system_dict["local"]["transforms_train"]["horizontal_flip"],
                            vertical_flip=system_dict["local"]["transforms_train"]["vertical_flip"],
                            validation_split=0.9,
                            rescale=0
                        );

    if(system_dict["local"]["transforms_train"]["featurewise_center"]):
        system_dict["local"]["data_generators"]["estimate"].mean = system_dict["local"]["transforms_train"]["mean"];
    if(system_dict["local"]["transforms_train"]["featurewise_std_normalization"]):
        system_dict["local"]["data_generators"]["estimate"].std = system_dict["local"]["transforms_train"]["std"];

    return system_dict;




@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_transform_trainval(system_dict):
    
    if("train-val" in system_dict["dataset"]["dataset_type"]):
        if(system_dict["local"]["transforms_train"]["featurewise_center"]):
            rescale_train = 1/256;
        else:
            rescale_train = 0;
        if(system_dict["local"]["transforms_val"]["featurewise_center"]):
            rescale_val = 1/256;
        else:
            rescale_val = 0;


        system_dict["local"]["data_generators"]["train"] = keras.preprocessing.image.ImageDataGenerator(
                            featurewise_center=system_dict["local"]["transforms_train"]["featurewise_center"],
                            featurewise_std_normalization=system_dict["local"]["transforms_train"]["featurewise_std_normalization"],
                            rotation_range=system_dict["local"]["transforms_train"]["rotation_range"],
                            width_shift_range=system_dict["local"]["transforms_train"]["width_shift_range"],
                            height_shift_range=system_dict["local"]["transforms_train"]["height_shift_range"],
                            shear_range=system_dict["local"]["transforms_train"]["shear_range"],
                            zoom_range=system_dict["local"]["transforms_train"]["zoom_range"],
                            brightness_range=system_dict["local"]["transforms_train"]["brightness_range"],
                            horizontal_flip=system_dict["local"]["transforms_train"]["horizontal_flip"],
                            vertical_flip=system_dict["local"]["transforms_train"]["vertical_flip"],
                            rescale = 0
                            );
                                       

       



        system_dict["local"]["data_generators"]["val"] = keras.preprocessing.image.ImageDataGenerator(
                            featurewise_center=system_dict["local"]["transforms_val"]["featurewise_center"],
                            featurewise_std_normalization=system_dict["local"]["transforms_val"]["featurewise_std_normalization"],
                            rotation_range=system_dict["local"]["transforms_val"]["rotation_range"],
                            width_shift_range=system_dict["local"]["transforms_val"]["width_shift_range"],
                            height_shift_range=system_dict["local"]["transforms_val"]["height_shift_range"],
                            shear_range=system_dict["local"]["transforms_val"]["shear_range"],
                            zoom_range=system_dict["local"]["transforms_val"]["zoom_range"],
                            brightness_range=system_dict["local"]["transforms_val"]["brightness_range"],
                            horizontal_flip=system_dict["local"]["transforms_val"]["horizontal_flip"],
                            vertical_flip=system_dict["local"]["transforms_val"]["vertical_flip"],
                            rescale = 0
                        );


        if(system_dict["local"]["transforms_train"]["featurewise_center"]):
            system_dict["local"]["data_generators"]["train"].mean = system_dict["local"]["transforms_train"]["mean"];
        if(system_dict["local"]["transforms_val"]["featurewise_center"]):
            system_dict["local"]["data_generators"]["val"].mean = system_dict["local"]["transforms_val"]["mean"];
        if(system_dict["local"]["transforms_train"]["featurewise_std_normalization"]):
            system_dict["local"]["data_generators"]["train"].std = system_dict["local"]["transforms_train"]["std"];
        if(system_dict["local"]["transforms_val"]["featurewise_std_normalization"]):
            system_dict["local"]["data_generators"]["val"].std = system_dict["local"]["transforms_val"]["std"];




    else:
        if(system_dict["local"]["transforms_train"]["featurewise_center"]):
            rescale_train = 1/256;
        else:
            rescale_train = 0;


        system_dict["local"]["data_generators"]["train"] = keras.preprocessing.image.ImageDataGenerator(
                            featurewise_center=system_dict["local"]["transforms_train"]["featurewise_center"],
                            featurewise_std_normalization=system_dict["local"]["transforms_train"]["featurewise_std_normalization"],
                            rotation_range=system_dict["local"]["transforms_train"]["rotation_range"],
                            width_shift_range=system_dict["local"]["transforms_train"]["width_shift_range"],
                            height_shift_range=system_dict["local"]["transforms_train"]["height_shift_range"],
                            shear_range=system_dict["local"]["transforms_train"]["shear_range"],
                            zoom_range=system_dict["local"]["transforms_train"]["zoom_range"],
                            brightness_range=system_dict["local"]["transforms_train"]["brightness_range"],
                            horizontal_flip=system_dict["local"]["transforms_train"]["horizontal_flip"],
                            vertical_flip=system_dict["local"]["transforms_train"]["vertical_flip"],
                            validation_split=1-system_dict["dataset"]["params"]["train_val_split"],
                            rescale = 0
                            );


        if(system_dict["local"]["transforms_train"]["featurewise_center"]):
            system_dict["local"]["data_generators"]["train"].mean = system_dict["local"]["transforms_train"]["mean"];
        if(system_dict["local"]["transforms_train"]["featurewise_std_normalization"]):
            system_dict["local"]["data_generators"]["train"].std = system_dict["local"]["transforms_train"]["std"];




    return system_dict;


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_transform_test(system_dict):
    if(system_dict["local"]["transforms_test"]["featurewise_center"]):
        rescale_val = 1/256;
    else:
        rescale_val = 0;

    system_dict["local"]["data_generators"]["test"] = keras.preprocessing.image.ImageDataGenerator(
                            featurewise_center=system_dict["local"]["transforms_test"]["featurewise_center"],
                            featurewise_std_normalization=system_dict["local"]["transforms_test"]["featurewise_std_normalization"],
                            rotation_range=system_dict["local"]["transforms_test"]["rotation_range"],
                            width_shift_range=system_dict["local"]["transforms_test"]["width_shift_range"],
                            height_shift_range=system_dict["local"]["transforms_test"]["height_shift_range"],
                            shear_range=system_dict["local"]["transforms_test"]["shear_range"],
                            zoom_range=system_dict["local"]["transforms_test"]["zoom_range"],
                            brightness_range=system_dict["local"]["transforms_test"]["brightness_range"],
                            horizontal_flip=system_dict["local"]["transforms_test"]["horizontal_flip"],
                            vertical_flip=system_dict["local"]["transforms_test"]["vertical_flip"],
                            rescale = 0
                            );

    if(system_dict["local"]["transforms_test"]["featurewise_center"]):
        system_dict["local"]["data_generators"]["test"].mean = system_dict["local"]["transforms_test"]["mean"];
    if(system_dict["local"]["transforms_test"]["featurewise_std_normalization"]):
        system_dict["local"]["data_generators"]["test"].std = system_dict["local"]["transforms_test"]["std"];

    return system_dict;