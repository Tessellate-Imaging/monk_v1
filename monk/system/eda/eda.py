from system.eda.imports import *
from system.eda.utils import *
from system.graphs.bar import create_plot
from system.imports import *


@accepts(dict, bool, bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def class_imbalance(system_dict, show_img, save_img):
    dataset_type = system_dict["dataset"]["dataset_type"];
    dataset_train_path = system_dict["dataset"]["train_path"];
    dataset_val_path = system_dict["dataset"]["val_path"];
    delimiter = system_dict["dataset"]["params"]["delimiter"];
    log_dir = system_dict["log_dir"];

    if("csv" in dataset_type):
        csv_train = system_dict["dataset"]["csv_train"];
        csv_val = system_dict["dataset"]["csv_val"];

    if(dataset_type == "train"):
        classes_folder, classes_folder_strength = populate_from_folder_train(dataset_train_path);                   
    elif(dataset_type == "train-val"):
        classes_folder, classes_folder_strength = populate_from_folder_trainval(dataset_train_path, dataset_val_path);

    elif(dataset_type == "csv_train"):
        classes_folder, classes_folder_strength = populate_from_csv_train(csv_train, delimiter);

    elif(dataset_type == "csv_train-val"):
        classes_folder, classes_folder_strength = populate_from_csv_trainval(csv_train, csv_val, delimiter);

    create_plot(classes_folder, classes_folder_strength, log_dir, show_img, save_img);

    return classes_folder, classes_folder_strength;


@accepts(dict, bool, bool, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def corrupted_missing_images(system_dict, check_missing, check_corrupt):
    dataset_type = system_dict["dataset"]["dataset_type"];
    dataset_train_path = system_dict["dataset"]["train_path"];
    dataset_val_path = system_dict["dataset"]["val_path"];

    if("csv" in dataset_type):
        csv_train = system_dict["dataset"]["csv_train"];
        csv_val = system_dict["dataset"]["csv_val"];
        delimiter = system_dict["dataset"]["params"]["delimiter"];

    missing_images_train = None;
    missing_images_val = None;
    corrupt_images_train = None;
    corrupt_images_val = None;



    if(dataset_type == "train"):
        if(check_missing):
            x=0;
        if(check_corrupt):
            corrupt_images_train = populate_corrupt_from_foldered(dataset_train_path, verbose=system_dict["verbose"]);
    elif(dataset_type == "train-val"):
        if(check_missing):
            x=0;
        if(check_corrupt):
            corrupt_images_train = populate_corrupt_from_foldered(dataset_train_path, verbose=system_dict["verbose"])
            corrupt_images_val = populate_corrupt_from_foldered(dataset_val_path, verbose=system_dict["verbose"])
    elif(dataset_type == "csv_train"):
        if(check_missing):
            missing_images_train = populate_missing(csv_train, dataset_train_path, delimiter);
        if(check_corrupt):
            corrupt_images_train = populate_corrupt_from_csv(csv_train, dataset_train_path, delimiter, verbose=system_dict["verbose"]);
    elif(dataset_type == "csv_train-val"):
        if(check_missing):
            missing_images_train = populate_missing(csv_train, dataset_train_path, delimiter);
            missing_images_val = populate_missing(csv_val, dataset_val_path, delimiter);
        if(check_corrupt):
            corrupt_images_train = populate_corrupt_from_csv(csv_train, dataset_train_path, delimiter, verbose=system_dict["verbose"]);
            corrupt_images_val = populate_corrupt_from_csv(csv_val, dataset_val_path, delimiter, verbose=system_dict["verbose"]);

    return missing_images_train, missing_images_val, corrupt_images_train, corrupt_images_val;