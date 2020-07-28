from monk.pytorch.finetune.imports import *
from monk.system.imports import *

from monk.pytorch.finetune.level_12_losses_main import prototype_losses


class prototype_updates(prototype_losses):
    '''
    Main class for all parametric update functions

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    @accepts("self", verbose=int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    ##########################################################################################################################################################
    @warning_checks(None, ["gte", 32, "lte", 1024], post_trace=False)
    @error_checks(None, ["gt", 0], post_trace=False)
    @accepts("self", int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_input_size(self, input_size):
        '''
        Update input size.

        Args:
            input_size (int): New input size

        Returns:
            None
        '''
        self.system_dict = set_input_size(input_size, self.system_dict);
        
        self.custom_print("Update: Input size - {}".format(self.system_dict["dataset"]["params"]["input_size"]));
        self.custom_print("");
        

    @warning_checks(None, ["lte", 128], post_trace=False)
    @error_checks(None, ["gt", 0], post_trace=False)
    @accepts("self", int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_batch_size(self, batch_size):
        '''
        Update batch size.

        Args:
            batch_size (int): New batch size

        Returns:
            None
        '''
        self.system_dict = set_batch_size(batch_size, self.system_dict);
        
        self.custom_print("Update: Batch size - {}".format(self.system_dict["dataset"]["params"]["batch_size"]));
        self.custom_print("");
        

    @accepts("self", bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_shuffle_data(self, shuffle):
        '''
        Update to shuffle data or not.

        Args:
            shuffle (bool): If True, will shuffle data

        Returns:
            None
        '''
        self.system_dict = set_data_shuffle(shuffle, self.system_dict);
        
        self.custom_print("Update: Data shuffle - {}".format(self.system_dict["dataset"]["params"]["train_shuffle"]));
        self.custom_print("");
        
    
    @warning_checks(None, ["lte", psutil.cpu_count()], post_trace=False)
    @error_checks(None, ["gt", 0], post_trace=False)
    @accepts("self", int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True) 
    def update_num_processors(self, num_processors):
        '''
        Update num processors for data loader.

        Args:
            num_processors (int): Max CPUs for data sampling

        Returns:
            None
        '''
        self.system_dict = set_num_processors(num_processors, self.system_dict);
        
        self.custom_print("Update: Num processors - {}".format(self.system_dict["dataset"]["params"]["num_workers"]));
        self.custom_print("");
        

    @accepts("self", bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_weighted_sampling(self, sample):
        '''
        Function inactive
        '''
        self.system_dict = set_weighted_sampling(sample, self.system_dict);
        
        self.custom_print("Update: Weighted Sampling - {}".format(self.system_dict["dataset"]["params"]["weighted_sample"]));
        self.custom_print("");



    @warning_checks(None, ["gt", 0.5, "lt", 1], post_trace=False)
    @error_checks(None, ["gt", 0, "lt", 1], post_trace=False)
    @accepts("self", float, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_trainval_split(self, value):
        '''
        Update training-validation split
        Args:
            split (float): Indicating train validation split
                            Division happens as follows:
                                train - total dataset * split * 100
                                val   - total dataset * (1-split) * 100 

        Returns:
            None
        '''
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


    @warning_checks(None, dataset_path=None, split=["gt", 0.5, "lt", 1], path_to_csv=None, delimiter=None, post_trace=False)
    @error_checks(None, dataset_path=["folder", 'r'], split=["gt", 0, "lt", 1], path_to_csv=["file", 'r'], delimiter=["in", [",", ";", "-", " "]], post_trace=False)
    @accepts("self", dataset_path=[str, list], split=float, path_to_csv=[str, list, bool], delimiter=str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_dataset(self, dataset_path=False, split=0.9, path_to_csv=False, delimiter=","):
        '''
        Update dataset path

        Args:
            dataset_path (str, list): Path to Dataset folder
                                      1) Single string if validation data does not exist
                                      2) List [train_path, val_path] in case of separate train and val data
            path_to_csv (str, list): Path to csv file pointing towards images
                                     1) Single string if validation data does not exist
                                     2) List [train_path, val_path] in case of separate train and val data
            value (float): Indicating train validation split
                            Division happens as follows:
                                train - total dataset * split * 100
                                val   - total dataset * (1-split) * 100 
            delimiter (str): Delimiter for csv file

        Returns:
            None
        '''
        self.system_dict = set_dataset_train_path(self.system_dict, dataset_path, split, path_to_csv, delimiter);
    ##########################################################################################################################################################




    ##########################################################################################################################################################
    @accepts("self", str, force=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_model_name(self, model_name, force=False):
        '''
        Update model name

        Args:
            model_name (str): Select from available models. Check via List_Models() function
            force (bool): Dummy function

        Returns:
            None
        '''
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
    ##########################################################################################################################################################



    ##########################################################################################################################################################
    @accepts("self", [str, list], force=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_model_path(self, model_path, force=False):
        '''
        Update model path for inferencing

        Args:
            model_path (str): Path to model weights.
            force (bool): Dummy function

        Returns:
            None
        '''
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
    ##########################################################################################################################################################
        


    ##########################################################################################################################################################
    @accepts("self", bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_use_gpu(self, gpu):
        '''
        Update to use gpu or cpu

        Args:
            gpu (bool): If True, then use GPU

        Returns:
            None
        '''
        self.system_dict = set_device(gpu, self.system_dict);
        
        self.custom_print("Update: Use Gpu - {}".format(self.system_dict["model"]["params"]["use_gpu"]));
        self.custom_print("");
    ##########################################################################################################################################################
        


    ##########################################################################################################################################################
    @accepts("self", bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_use_pretrained(self, pretrained):
        '''
        Update to use pretrained wights or randomly initialized weights

        Args:
            pretrained (bool): If True, use pretrained weights
                                else, use randomly initialized weights

        Returns:
            None
        '''
        self.system_dict = set_pretrained(pretrained, self.system_dict);
        
        self.custom_print("Update: Use pretrained - {}".format(self.system_dict["model"]["params"]["use_pretrained"]));
        self.custom_print("");
    ##########################################################################################################################################################
        


    ##########################################################################################################################################################
    @accepts("self", bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_freeze_base_network(self, freeze):
        '''
        Update whether freeze base network or not

        Args:
            freeze (bool): If True, then base network is non-trainable, works as a feature extractor

        Returns:
            None
        '''
        self.system_dict = set_freeze_base_network(freeze, self.system_dict);
        
        self.custom_print("Update: Freeze Base Network - {}".format(self.system_dict["model"]["params"]["freeze_base_network"]));
        self.custom_print("");
    ##########################################################################################################################################################


    ##########################################################################################################################################################
    @error_checks(None, ["gte", 0], post_trace=False)
    @accepts("self", int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_freeze_layers(self, num_freeze):
        '''
        Update to freeze certain layers in the network

        Args:
            num_freeze (int): Number of layers to freeze in network starting from top

        Returns:
            None
        '''
        self.system_dict["model"]["params"]["num_freeze"] = num_freeze;
        
        self.custom_print("Update: Freeze layers - {}".format(self.system_dict["model"]["params"]["num_freeze"]));
        self.custom_print("");
    ##########################################################################################################################################################




    ##########################################################################################################################################################
    @warning_checks(None, ["lt", 100], post_trace=False)
    @error_checks(None, ["gt", 0], post_trace=False)
    @accepts("self", int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_num_epochs(self, num_epochs):
        '''
        Update number of epochs to train the network

        Args:
            num_epochs (int): New number of epochs

        Returns:
            None
        '''
        self.system_dict = set_num_epochs(num_epochs, self.system_dict);
        
        self.custom_print("Update: Num Epochs - {}".format(self.system_dict["hyper-parameters"]["num_epochs"]));
        self.custom_print("");
    ##########################################################################################################################################################


    ##########################################################################################################################################################
    @warning_checks(None, ["lt", 1], post_trace=False)
    @error_checks(None, ["gt", 0], post_trace=False)
    @accepts("self", [int, float], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_learning_rate(self, learning_rate):
        '''
        Update base learning rate for training

        Args:
            learning_rate (float): New base learning rate
            
        Returns:
            None
        '''
        self.system_dict["hyper-parameters"]["learning_rate"] = learning_rate;
        self.system_dict["hyper-parameters"]["optimizer"]["params"]["lr"] = learning_rate;

        self.custom_print("Update: Learning Rate - {}".format(self.system_dict["hyper-parameters"]["learning_rate"]));
        self.custom_print("");
    ##########################################################################################################################################################



    ##########################################################################################################################################################
    @accepts("self", bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_display_progress_realtime(self, value):    
        '''
        Update display progress param

        Args:
            value (bool): If True, then real time progress is displayed
            
        Returns:
            None
        ''' 
        self.system_dict = set_display_progress_realtime(value, self.system_dict);
        
        self.custom_print("Update: Display progress realtime - {}".format(self.system_dict["training"]["settings"]["display_progress_realtime"]));
        self.custom_print("");
    ##########################################################################################################################################################
        

    ##########################################################################################################################################################
    @accepts("self", bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_display_progress(self, value): 
        '''
        Update display progress param

        Args:
            value (bool): If True, then per epoch progress is displayed
            
        Returns:
            None
        ''' 
        self.system_dict = set_display_progress(value, self.system_dict);
        
        self.custom_print("Update: Display progress  - {}".format(self.system_dict["training"]["settings"]["display_progress"]));
        self.custom_print("");
    ##########################################################################################################################################################
        

    ##########################################################################################################################################################
    @error_checks(None, None, prefix=["name", ["A-Z", "a-z", "0-9", "-", "_"]], post_trace=False)
    @accepts("self", bool, prefix=str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_save_intermediate_models(self, value, prefix="intermediate_model_"): 
        '''
        Update whether to save intermediate models or not

        Args:
            value (bool): If True, saves model weight post every epoch
            prefix (str): Appends a prefix to intermediate weights
            
        Returns:
            None
        '''
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
    ##########################################################################################################################################################
        

    ##########################################################################################################################################################
    @accepts("self", bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def update_save_training_logs(self, value):
        '''
        Update whether to save training logs or not

        Args:
            value (bool): If True, saves all training and validation metrics. Required for comparison.
            
        Returns:
            None
        '''
        self.system_dict = set_save_training_logs(value, self.system_dict);
        
        self.custom_print("Update: Save Training logs - {}".format(self.system_dict["training"]["settings"]["save_training_logs"]));
        self.custom_print("");
    ##########################################################################################################################################################