from pytorch.datasets.imports import *
from system.imports import *



@accepts(dict, [str, list, bool], [float, int, bool], [str, list, bool], str, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_dataset_train_path(system_dict, path, split, path_to_csv, delimiter):
    dataset_type = None;
    dataset_train_path = None;
    dataset_val_path = None;
    csv_train = None;
    csv_val = None;
    train_val_split = None;
    if(path_to_csv):
        if(type(path) == str):
            dataset_type = "csv_train";
            csv_train = path_to_csv;
            dataset_train_path = path;
            train_val_split = split;
        elif(type(path) == list):
            dataset_type = "csv_train-val";
            csv_train = path_to_csv[0];
            csv_val = path_to_csv[1];
            dataset_train_path = path[0];
            dataset_val_path = path[1];
            train_val_split = None;
    else:
        if(type(path) == str):
            dataset_type = "train";
            dataset_train_path = path;
            train_val_split = split;
        elif(type(path) == list):
            dataset_type = "train-val";
            dataset_train_path = path[0];
            dataset_val_path = path[1];
            train_val_split = None;

    system_dict["dataset"]["dataset_type"] = dataset_type;
    system_dict["dataset"]["train_path"] = dataset_train_path;
    system_dict["dataset"]["val_path"] = dataset_val_path;
    system_dict["dataset"]["csv_train"] = csv_train;
    system_dict["dataset"]["csv_val"] = csv_val;
    system_dict["dataset"]["params"]["train_val_split"] = train_val_split;
    system_dict["dataset"]["params"]["delimiter"] = delimiter;

    return system_dict;


@accepts(dict, [str, bool], [str, bool], str, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def set_dataset_test_path(system_dict,  path, path_to_csv, delimiter):
    dataset_test_type = None;
    dataset_test_path = None;
    csv_test = None;
    if(path_to_csv):
        csv_test = path_to_csv;
        dataset_test_path = path;
        dataset_test_type = "csv";
    else:
        dataset_test_path = path;
        dataset_test_type = "foldered";
    
    system_dict["dataset"]["test_path"] = dataset_test_path;
    system_dict["dataset"]["csv_test"] = csv_test;
    system_dict["dataset"]["params"]["test_delimiter"] = delimiter;
    system_dict["dataset"]["params"]["dataset_test_type"] = dataset_test_type;


    return system_dict;