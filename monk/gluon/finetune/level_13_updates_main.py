from gluon.finetune.imports import *
from system.imports import *

from gluon.finetune.level_12_losses_main import prototype_losses


class prototype_updates(prototype_losses):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    ##########################################################################################################################################################
    @warning_checks(None, ["gte", 32, "lte", 1024], post_trace=True)
    @error_checks(None, ["gt", 0], post_trace=True)
    @accepts("self", int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_input_size(self, input_size):
        self.system_dict = set_input_size(input_size, self.system_dict);
        
        self.custom_print("Update: Input size - {}".format(self.system_dict["dataset"]["params"]["input_size"]));
        self.custom_print("");
        

    @warning_checks(None, ["lte", 128], post_trace=True)
    @error_checks(None, ["gt", 0], post_trace=True)
    @accepts("self", int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_batch_size(self, batch_size):
        self.system_dict = set_batch_size(batch_size, self.system_dict);
        
        self.custom_print("Update: Batch size - {}".format(self.system_dict["dataset"]["params"]["batch_size"]));
        self.custom_print("");
        

    @accepts("self", bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_shuffle_data(self, shuffle):
        self.system_dict = set_data_shuffle(shuffle, self.system_dict);
        
        self.custom_print("Update: Data shuffle - {}".format(self.system_dict["dataset"]["params"]["train_shuffle"]));
        self.custom_print("");
        
    
    @warning_checks(None, ["lte", psutil.cpu_count()], post_trace=True)
    @error_checks(None, ["gt", 0], post_trace=True)
    @accepts("self", int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True) 
    def update_num_processors(self, num_processors):
        self.system_dict = set_num_processors(num_processors, self.system_dict);
        
        self.custom_print("Update: Num processors - {}".format(self.system_dict["dataset"]["params"]["num_workers"]));
        self.custom_print("");
        

    @accepts("self", bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_weighted_sampling(self, sample):
        self.system_dict = set_weighted_sampling(sample, self.system_dict);
        
        self.custom_print("Update: Weighted Sampling - {}".format(self.system_dict["dataset"]["params"]["weighted_sample"]));
        self.custom_print("");



    @warning_checks(None, ["gt", 0.5, "lt", 1], post_trace=True)
    @error_checks(None, ["gt", 0, "lt", 1], post_trace=True)
    @accepts("self", float, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_trainval_split(self, value):
        if(self.system_dict["dataset"]["dataset_type"] == "train"):
            dataset_path = self.system_dict["dataset"]["train_path"];
            path_to_csv=False;
        elif(self.system_dict["dataset"]["dataset_type"] == "train-val"):
            dataset_path = [self.system_dict["dataset"]["train_path"], self.system_dict["dataset"]["val_path"]];
            path_to_csv=False;
        elif(self.system_dict["dataset"]["dataset_type"] == "csv_train"):
            dataset_path = self.system_dict["dataset"]["train_path"];
            path_to_csv = self.system_dict["dataset"]["csv_train"];
        elif(self.system_dict["dataset"]["dataset_type"] == "csv_train-val"):
            dataset_path = [self.system_dict["dataset"]["train_path"], self.system_dict["dataset"]["val_path"]];
            path_to_csv = [self.system_dict["dataset"]["csv_train"], self.system_dict["dataset"]["csv_val"]];
        else:
            msg = "Dataset Type invalid.\n";
            msg += "Cannot update split"
            ConstraintsWarning(msg)

        self.system_dict = set_dataset_train_path(self.system_dict, dataset_path, value, path_to_csv, self.system_dict["dataset"]["params"]["delimiter"]);


    @warning_checks(None, dataset_path=None, split=["gt", 0.5, "lt", 1], path_to_csv=None, delimiter=None, post_trace=True)
    @error_checks(None, dataset_path=["folder", 'r'], split=["gt", 0, "lt", 1], path_to_csv=["file", 'r'], delimiter=["in", [",", ";", "-", " "]], post_trace=True)
    @accepts("self", dataset_path=[str, list], split=float, path_to_csv=[str, list, bool], delimiter=str, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_dataset(self, dataset_path=False, split=0.9, path_to_csv=False, delimiter=","):
        self.system_dict = set_dataset_train_path(self.system_dict, dataset_path, split, path_to_csv, delimiter);
    ##########################################################################################################################################################




    ##########################################################################################################################################################
    @accepts("self", str, force=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_model_name(self, model_name, force=False):
        if(not force):
            if(self.system_dict["training"]["status"]):
                ConstraintWarning("Model trained using {}\n".format(self.system_dict["model"]["params"]["model_name"]));
                ConstraintWarning("Changing the model will overwrite previously trained models if training is executed.\n");
                inp = input("Do you wish to continue further (y/n):");

                if(inp == "y"):
                    self.system_dict = set_model_name(model_name, self.system_dict);
                    self.custom_print("Update: Model name - {}".format(self.system_dict["model"]["params"]["model_name"]));
                    self.custom_print("");
                else:
                    self.custom_print("Model not updated.");
                    self.custom_print("");
            else:
                self.system_dict = set_model_name(model_name, self.system_dict);
                self.custom_print("Update: Model name - {}".format(self.system_dict["model"]["params"]["model_name"]));
                self.custom_print("");
        else:
            self.system_dict = set_model_name(model_name, self.system_dict);
            self.custom_print("Update: Model name - {}".format(self.system_dict["model"]["params"]["model_name"]));
            self.custom_print("");

            

            


    @accepts("self", [str, list], force=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_model_path(self, model_path, force=False):
        if(not force):
            if(self.system_dict["training"]["status"]):
                ConstraintWarning("Model trained using {}\n".format(self.system_dict["model"]["params"]["model_name"]));
                ConstraintWarning("Changing the model will overwrite previously trained models if training is executed.\n");
                inp = input("Do you wish to continue further (y/n):");

                if(inp == "y"):
                    self.system_dict = set_model_path(model_path, self.system_dict);
                    self.custom_print("Update: Model path - {}".format(self.system_dict["model"]["params"]["model_path"]));
                    self.custom_print("");
                else:
                    self.custom_print("Model not updated.");
                    self.custom_print("");
            else:
                self.system_dict = set_model_path(model_path, self.system_dict);
                self.custom_print("Update: Model path - {}".format(self.system_dict["model"]["params"]["model_path"]));
                self.custom_print("");
        else:
            self.system_dict = set_model_path(model_path, self.system_dict);
            self.custom_print("Update: Model path - {}".format(self.system_dict["model"]["params"]["model_path"]));
            self.custom_print("");               
        

    @accepts("self", bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_use_gpu(self, gpu):
        self.system_dict = set_device(gpu, self.system_dict);
        
        self.custom_print("Update: Use Gpu - {}".format(self.system_dict["model"]["params"]["use_gpu"]));
        self.custom_print("");
        

    @accepts("self", bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_use_pretrained(self, pretrained):
        self.system_dict = set_pretrained(pretrained, self.system_dict);
        
        self.custom_print("Update: Use pretrained - {}".format(self.system_dict["model"]["params"]["use_pretrained"]));
        self.custom_print("");
        


    @accepts("self", bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_freeze_base_network(self, freeze):
        self.system_dict = set_freeze_base_network(freeze, self.system_dict);
        
        self.custom_print("Update: Freeze Base Network - {}".format(self.system_dict["model"]["params"]["freeze_base_network"]));
        self.custom_print("");
        


    @error_checks(None, ["gte", 0], post_trace=True)
    @accepts("self", int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_freeze_layers(self, num_freeze):
        self.system_dict["model"]["params"]["num_freeze"] = num_freeze;
        
        self.custom_print("Update: Freeze layers - {}".format(self.system_dict["model"]["params"]["num_freeze"]));
        self.custom_print("");
    ##########################################################################################################################################################





    ##########################################################################################################################################################
    @warning_checks(None, ["lt", 100], post_trace=True)
    @error_checks(None, ["gt", 0], post_trace=True)
    @accepts("self", int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_num_epochs(self, num_epochs):
        self.system_dict = set_num_epochs(num_epochs, self.system_dict);
        
        self.custom_print("Update: Num Epochs - {}".format(self.system_dict["hyper-parameters"]["num_epochs"]));
        self.custom_print("");


    @warning_checks(None, ["lt", 1], post_trace=True)
    @error_checks(None, ["gt", 0], post_trace=True)
    @accepts("self", [int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_learning_rate(self, learning_rate):
        self.system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
        self.system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;
        
        self.custom_print("Update: Learning Rate - {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("");
        

    @accepts("self", bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_display_progress_realtime(self, value):    
        self.system_dict = set_display_progress_realtime(value, self.system_dict);
        
        self.custom_print("Update: Display progress realtime - {}".format(self.system_dict["training"]["settings"]["display_progress_realtime"]));
        self.custom_print("");
        

    @accepts("self", bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_display_progress(self, value): 
        self.system_dict = set_display_progress(value, self.system_dict);
        
        self.custom_print("Update: Display progress  - {}".format(self.system_dict["training"]["settings"]["display_progress"]));
        self.custom_print("");
        

    @error_checks(None, None, prefix=["name", ["A-Z", "a-z", "0-9", "-", "_"]], post_trace=True)
    @accepts("self", bool, prefix=str, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_save_intermediate_models(self, value, prefix="intermediate_model_"): 
        if(value):
            if(not os.access(self.system_dict["model_dir"], os.W_OK)):
                msg = "Folder \"{}\" has no read access".format(self.system_dict["model_dir"])
                msg += "Cannot save Intermediate models";
                raise ConstraintError(msg);

        self.system_dict = set_save_intermediate_models(value, self.system_dict);
        self.system_dict = set_intermediate_model_prefix(prefix, self.system_dict);
        
        self.custom_print("Update: Save Intermediate models - {}".format(self.system_dict["training"]["settings"]["save_intermediate_models"]));
        if(self.system_dict["training"]["settings"]["save_intermediate_models"]):
            self.custom_print("Update: Intermediate model prefix - {}".format(self.system_dict["training"]["settings"]["intermediate_model_prefix"]));
        self.custom_print("");
        

    @accepts("self", bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def update_save_training_logs(self, value):
        self.system_dict = set_save_training_logs(value, self.system_dict);
        
        self.custom_print("Update: Save Training logs - {}".format(self.system_dict["training"]["settings"]["save_training_logs"]));
        self.custom_print("");
    ##########################################################################################################################################################
