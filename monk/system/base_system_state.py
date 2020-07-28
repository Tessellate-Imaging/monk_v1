from monk.system.imports import *


@accepts(post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=True)
def get_base_system_dict():
    system_dict = {};

    system_dict["verbose"] = 1;
    system_dict["cwd"] = False;
    system_dict["master_systems_dir"] = False
    system_dict["master_systems_dir_relative"] = False;


    system_dict["project_name"] = False;
    system_dict["project_dir"] = False;
    system_dict["project_dir_relative"] = False;
    system_dict["experiment_name"] = False;
    system_dict["experiment_dir"] = False;
    system_dict["experiment_dir_relative"] = False;
    system_dict["origin"] = False;

    system_dict["master_comparison_dir"] = False;
    system_dict["master_comparison_dir_relative"] = False;
    system_dict["library"] = False;
    system_dict["output_dir"] = False;
    system_dict["output_dir_relative"] = False;
    system_dict["model_dir"] = False;
    system_dict["model_dir_relative"] = False;
    system_dict["log_dir"] = False;
    system_dict["log_dir_relative"] = False;
    system_dict["fname"] = False;
    system_dict["fname_relative"] = False;


    #Dataset details
    system_dict["dataset"] = {};
    system_dict["dataset"]["dataset_type"] = False
    system_dict["dataset"]["label_type"] = False;
    system_dict["dataset"]["train_path"] = False;
    system_dict["dataset"]["val_path"] = False;
    system_dict["dataset"]["csv_train"] = False;
    system_dict["dataset"]["csv_val"] = False;
    system_dict["dataset"]["test_path"] = False;
    system_dict["dataset"]["csv_test"] = False;


    #Dataset params
    system_dict["dataset"]["params"] = {};
    system_dict["dataset"]["params"]["input_size"] = False;
    system_dict["dataset"]["params"]["data_shape"] = False;
    system_dict["dataset"]["params"]["batch_size"] = False;
    system_dict["dataset"]["params"]["train_shuffle"] = False;
    system_dict["dataset"]["params"]["train_shuffle"] = False;
    system_dict["dataset"]["params"]["num_workers"] = False;
    system_dict["dataset"]["params"]["weighted_sample"] = False;
    system_dict["dataset"]["params"]["num_classes"] = False;
    system_dict["dataset"]["params"]["classes"] = False;
    system_dict["dataset"]["params"]["num_train_images"] = False;
    system_dict["dataset"]["params"]["num_val_images"] = False;
    system_dict["dataset"]["params"]["num_test_images"] = False;
    system_dict["dataset"]["params"]["delimiter"] = ",";
    system_dict["dataset"]["params"]["test_delimiter"] = ",";
    system_dict["dataset"]["params"]["dataset_test_type"] = False;
    system_dict["dataset"]["params"]["train_val_split"] = 0.9;


    #Dataset transforms
    system_dict["dataset"]["transforms"] = {};
    system_dict["dataset"]["transforms"]["train"] = [];
    system_dict["dataset"]["transforms"]["val"] = [];
    system_dict["dataset"]["transforms"]["test"] = [];
    system_dict["dataset"]["status"] = False;

    #Model details
    system_dict["model"] = {};
    system_dict["model"]["status"] = False;
    system_dict["model"]["final_layer"] = False;
    system_dict["model"]["type"] = "pretrained";
    system_dict["model"]["custom_network"] = [];

    #Custom Model details
    system_dict["custom_model"] = {};
    system_dict["custom_model"]["status"] = False;
    system_dict["custom_model"]["network_stack"] = [];
    system_dict["custom_model"]["network_initializer"] = False;

    
    #Model params
    system_dict["model"]["params"] = {};
    system_dict["model"]["params"]["model_name"] = False;
    system_dict["model"]["params"]["model_path"] = False;
    system_dict["model"]["params"]["use_gpu"] = False;
    system_dict["model"]["params"]["use_pretrained"] = False;
    system_dict["model"]["params"]["freeze_base_network"] = False;
    system_dict["model"]["params"]["num_layers"] = False;
    system_dict["model"]["params"]["num_params_to_update"] = False;
    system_dict["model"]["params"]["num_freeze"] = False;
    system_dict["model"]["params"]["gpu_memory_fraction"] = 0.6;



    #Hyper parameter details
    system_dict["hyper-parameters"] = {};
    system_dict["hyper-parameters"]["status"] = False;
    system_dict["hyper-parameters"]["learning_rate"] = False;
    system_dict["hyper-parameters"]["num_epochs"] = False;
    system_dict["hyper-parameters"]["optimizer"] = {};
    system_dict["hyper-parameters"]["optimizer"]["name"] = False;
    system_dict["hyper-parameters"]["optimizer"]["params"] = {};
    system_dict["hyper-parameters"]["learning_rate_scheduler"] = {};
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] = False;
    system_dict["hyper-parameters"]["learning_rate_scheduler"]["params"] = {};
    system_dict["hyper-parameters"]["loss"] = {};
    system_dict["hyper-parameters"]["loss"]["name"] = False;
    system_dict["hyper-parameters"]["loss"]["params"] = {};

    #Training details
    system_dict["training"] = {};
    system_dict["training"]["settings"] = {};
    system_dict["training"]["settings"]["display_progress_realtime"] = False;
    system_dict["training"]["settings"]["display_progress"] = False;
    system_dict["training"]["settings"]["save_intermediate_models"] = False;
    system_dict["training"]["settings"]["save_training_logs"] = False;
    system_dict["training"]["settings"]["intermediate_model_prefix"] = False;
    system_dict["training"]["outputs"] = {};
    system_dict["training"]["outputs"]["max_gpu_memory_usage"] = 0;
    system_dict["training"]["outputs"]["best_val_acc"] = 0;
    system_dict["training"]["outputs"]["best_val_acc_epoch_num"] = 0;
    system_dict["training"]["outputs"]["epochs_completed"] = 0;
    system_dict["training"]["status"] = False;

    #Testing details
    system_dict["testing"] = {};
    system_dict["testing"]["status"] = False;
    system_dict["testing"]["num_images"] = False;
    system_dict["testing"]["num_correct_predictions"] = False;
    system_dict["testing"]["percentage_accuracy"] = False;
    system_dict["testing"]["class_accuracy"] = False;

    #States
    system_dict["states"] = {};
    system_dict["states"]["eval_infer"] = False;
    system_dict["states"]["resume_train"] = False;
    system_dict["states"]["copy_from"] = False;
    system_dict["states"]["pseudo_copy_from"] = False;

    #Local variables
    system_dict["local"] = {};

    system_dict["local"]["projects_list"] = [];
    system_dict["local"]["num_projects"] = False;
    system_dict["local"]["experiments_list"] = [];
    system_dict["local"]["num_experiments"] = False;
    system_dict["local"]["project_experiment_list"] = [];

    system_dict["local"]["transforms_train"] = [];
    system_dict["local"]["transforms_val"] = [];
    system_dict["local"]["transforms_test"] = [];
    system_dict["local"]["normalize"] = False;
    system_dict["local"]["mean_subtract"] = False;
    system_dict["local"]["applied_train_tensor"] = False;
    system_dict["local"]["applied_test_tensor"] = False;
    system_dict["local"]["data_transforms"] = {};
    system_dict["local"]["image_datasets"] = {};
    system_dict["local"]["data_loaders"] = {};
    system_dict["local"]["data_generators"] = {};

    system_dict["local"]["model"] = False;
    system_dict["local"]["custom_model"] = False;
    system_dict["local"]["ctx"] = False;
    system_dict["local"]["params_to_update"] = [];
    system_dict["local"]["device"] = False;

    system_dict["local"]["learning_rate_scheduler"] = False;
    system_dict["local"]["optimizer"] = False;
    system_dict["local"]["criterion"] = False;

    #CNN Visualization storage details
    system_dict["visualization"] = {}
    
    system_dict["visualization"]["base"] = False;
    system_dict["visualization"]["kernels_dir"] = False;
    system_dict["visualization"]["feature_maps_dir"] = False;

    return system_dict;

#@accepts(post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def update_local_var(system_dict):
    system_dict["local"] = {};

    system_dict["local"]["projects_list"] = [];
    system_dict["local"]["num_projects"] = False;
    system_dict["local"]["experiments_list"] = [];
    system_dict["local"]["num_experiments"] = False;
    system_dict["local"]["project_experiment_list"] = [];

    system_dict["local"]["transforms_train"] = [];
    system_dict["local"]["transforms_val"] = [];
    system_dict["local"]["transforms_test"] = [];
    system_dict["local"]["normalize"] = False;
    system_dict["local"]["mean_subtract"] = False;
    system_dict["local"]["applied_train_tensor"] = False;
    system_dict["local"]["applied_test_tensor"] = False;
    system_dict["local"]["data_transforms"] = {};
    system_dict["local"]["image_datasets"] = {};
    system_dict["local"]["data_loaders"] = {};
    system_dict["local"]["data_generators"] = {};

    system_dict["local"]["model"] = False;
    system_dict["local"]["custom_model"] = False;
    system_dict["local"]["ctx"] = False;
    system_dict["local"]["params_to_update"] = [];
    system_dict["local"]["device"] = False;

    system_dict["local"]["learning_rate_scheduler"] = False;
    system_dict["local"]["optimizer"] = False;
    system_dict["local"]["criterion"] = False;


    system_dict["states"] = {};
    system_dict["states"]["eval_infer"] = False;
    system_dict["states"]["resume_train"] = False;
    system_dict["states"]["copy_from"] = False;
    system_dict["states"]["pseudo_copy_from"] = False;

    return system_dict;