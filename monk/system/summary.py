from system.imports import *
from system.common import read_json


@accepts(str, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=True)
def print_summary(fname):
    '''
    Read a system dictionary file and print summary

    Args:
        fname (str): Path to file 

    Returns:
        None
    '''
    system_dict = read_json(fname);

    #############################################################################################################################
    print("");
    print("");
    print("Experiment Summary");
    
    print("");
    print("System");
    print("    Project Name:    {}".format(system_dict["project_name"]));
    print("    Project Dir:     {}".format(system_dict["project_dir_relative"]));
    print("    Experiment Name: {}".format(system_dict["experiment_name"]));
    print("    Experiment Dir:  {}".format(system_dict["experiment_dir_relative"]));
    print("    Library:         {}".format(system_dict["library"]));
    print("    Origin:          {}".format(system_dict["origin"]));
    print("");
    #############################################################################################################################




    #############################################################################################################################
    print("Dataset");
    print("    Status:       {}".format(system_dict["dataset"]["status"]));
    if(system_dict["dataset"]["status"]):
        print("    Dataset Type: {}".format(system_dict["dataset"]["dataset_type"]));
        print("    Train path:   {}".format(system_dict["dataset"]["train_path"]));
        print("    Val path:     {}".format(system_dict["dataset"]["val_path"]));
        print("    Test path:    {}".format(system_dict["dataset"]["test_path"]));
        print("    CSV Train:    {}".format(system_dict["dataset"]["csv_train"]));
        print("    CSV Val:      {}".format(system_dict["dataset"]["csv_val"]));
        print("    CSV Test:     {}".format(system_dict["dataset"]["csv_test"]));
        print("");


        print("Dataset Parameters:");
        print("    Input Size:   {}".format(system_dict["dataset"]["params"]["input_size"]));
        print("    Batch Size:   {}".format(system_dict["dataset"]["params"]["batch_size"]));
        print("    Shuffle:      {}".format(system_dict["dataset"]["params"]["train_shuffle"]));
        print("    Processors:   {}".format(system_dict["dataset"]["params"]["num_workers"]));
        print("    Num Classes:  {}".format(system_dict["dataset"]["params"]["num_classes"]));
        print("");
        

        print("Dataset Transforms:");
        print("    Train transforms: {}".format(system_dict["dataset"]["transforms"]["train"]));
        print("    Val transforms:   {}".format(system_dict["dataset"]["transforms"]["val"]));
        print("    Test transforms:  {}".format(system_dict["dataset"]["transforms"]["test"]));
    print("");
    #############################################################################################################################



    #############################################################################################################################
    print("Model");
    print("    Status:".format(system_dict["model"]["status"]));
    if(system_dict["model"]["status"]):
        print("    Model Name:                     {}".format(system_dict["model"]["params"]["model_name"]));
        print("    Use Gpu:                        {}".format(system_dict["model"]["params"]["use_gpu"]));
        print("    Use pretrained weights:         {}".format(system_dict["model"]["params"]["use_pretrained"]));
        print("    Base network weights freezed:   {}".format(system_dict["model"]["params"]["freeze_base_network"]));
        print("    Number of trainable parameters: {}".format(system_dict["model"]["params"]["num_params_to_update"]));
    print("")
    #############################################################################################################################



    #############################################################################################################################
    print("Hyper-Parameters");
    print("    Status: {}".format(system_dict["hyper-parameters"]["status"]));
    if(system_dict["hyper-parameters"]["status"]):
        print("    Optimizer:                {}".format(system_dict["hyper-parameters"]["optimizer"]));
        print("    Learning Rate Scheduler:  {}".format(system_dict["hyper-parameters"]["learning_rate_scheduler"]));
        print("    loss:                     {}".format(system_dict["hyper-parameters"]["loss"]));
        print("    Num epochs:               {}".format(system_dict["hyper-parameters"]["num_epochs"]));
    print("");
    #############################################################################################################################



    #############################################################################################################################
    print("");
    print("Dataset Settings");
    if("display_progress" in system_dict["training"]["settings"].keys()):
        print("    Status: {}".format(True));
        print("    Display progress:          {}".format(system_dict["training"]["settings"]["display_progress"]));
        print("    Display progress realtime: {}".format(system_dict["training"]["settings"]["display_progress_realtime"]));
        print("    Save intermediate models:  {}".format(system_dict["training"]["settings"]["save_intermediate_models"]));
        print("    Save training logs:        {}".format(system_dict["training"]["settings"]["save_training_logs"]));
        print("    Intermediate model prefix: {}".format(system_dict["training"]["settings"]["intermediate_model_prefix"]));
    else:
        print("    Status: {}".format(False));
    print("");
    #############################################################################################################################



    #############################################################################################################################
    print("");
    print("Training");
    print("    Status: {}".format(system_dict["training"]["status"]));
    if(system_dict["training"]["status"]):
        print("    Model dir:                      {}".format(system_dict["model_dir_relative"]));
        print("    Best validation accuracy:       {}".format(system_dict["training"]["outputs"]["best_val_acc"]));
        print("    Best validation accuracy epoch: {}".format(system_dict["training"]["outputs"]["best_val_acc_epoch_num"]));
        print("    Training time:                  {}".format(system_dict["training"]["outputs"]["training_time"]));
        print("    Epochs completed:               {}".format(system_dict["training"]["outputs"]["epochs_completed"]));
        print("    Max Gpu Usage:                  {}".format(system_dict["training"]["outputs"]["max_gpu_usage"]));
        print("");

        print("Training Log Files");
        print("    Train accuracy: {}".format(system_dict["training"]["outputs"]["log_train_acc_history_relative"]));
        print("    Train loss:     {}".format(system_dict["training"]["outputs"]["log_train_loss_history_relative"]));
        print("    Val accuracy:   {}".format(system_dict["training"]["outputs"]["log_val_acc_history_relative"]));
        print("    Val loss:       {}".format(system_dict["training"]["outputs"]["log_val_loss_history_relative"]));
    print("");
    #############################################################################################################################


    
    #############################################################################################################################
    print("External Evaluation");
    print("    Status: {}".format(system_dict["testing"]["status"]));
    if(system_dict["testing"]["status"]):
        print("    Evaluation Dataset path: {}".format(system_dict["dataset"]["test_path"]));
        print("    Num Images:              {}".format(system_dict["testing"]["num_images"]));
        print("    Num correct predictions: {}".format(system_dict["testing"]["num_correct_predictions"]));
        print("    Overall Accuracy:        {} %".format(system_dict["testing"]["percentage_accuracy"]));
    print("");
    #############################################################################################################################

