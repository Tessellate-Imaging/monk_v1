from monk.gluon.finetune.imports import *
from monk.system.imports import *


from monk.gluon.finetune.level_6_params_main import prototype_params



class prototype_aux(prototype_params):
    '''
    Main class for all auxiliary functions - EDA, Estimate Training Time, Resetting params, switching modes, & debugging 

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    @accepts("self", verbose=int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);

    ###############################################################################################################################################
    def EDA(self, show_img=False, save_img=False, check_missing=False, check_corrupt=False):
        '''
        Experimental Data Analysis
            - Finding number of images in each class
            - Check missing images in case of csv type dataset
            - Find all corrupt images
        Args:
            show_img (bool): If True, displays bar graph for images per class 
            save_img (bool): If True, saves bar graph for images per class
            check_missing (bool): If True, checks for missing images in csv type dataset
            check_corrupt (bool): If True, checks for corrupted images in foldered and csv dataset

        Returns:
            None
        '''
        if(not self.system_dict["dataset"]["train_path"]):
            msg = "Dataset train path not set. Cannot run EDA";
            raise ConstraintError(msg);


        classes_folder, classes_folder_strength = class_imbalance(self.system_dict, show_img, save_img);
        missing_images_train, missing_images_val, corrupt_images_train, corrupt_images_val = corrupted_missing_images(self.system_dict, check_missing, check_corrupt);

        self.custom_print("EDA: Class imbalance")
        for i in range(len(classes_folder)):
            self.custom_print("    {}. Class: {}, Number: {}".format(i+1, classes_folder[i], classes_folder_strength[i]));
        self.custom_print("");

        if(check_missing):
            self.custom_print("EDA: Check Missing");
            if("csv" in self.system_dict["dataset"]["dataset_type"]):
                if(missing_images_train):
                    self.custom_print("    Missing Images in folder {}".format(self.system_dict["dataset"]["train_path"]));
                    for i in range(len(missing_images_train)):
                        self.custom_print("        {}. {}".format(i+1, missing_images_train[i]));
                    self.custom_print("");
                else:
                    self.custom_print("    All images present in train dir.");
                    self.custom_print("");

                if(missing_images_val):
                    self.custom_print("    Missing Images in folder {}".format(self.system_dict["dataset"]["val_path"]));
                    for i in range(len(missing_images_val)):
                        self.custom_print("        {}. {}".format(i+1, missing_images_val[i]));
                    self.custom_print("");
                else:
                    self.custom_print("    All images present in val dir.");
                    self.custom_print("");
            else:
                self.custom_print("    Missing check not required for foldered dataset");
                self.custom_print("");
            

        if(check_corrupt):
            self.custom_print("EDA: Check Corrupt");
            if(corrupt_images_train):
                self.custom_print("    Corrupt Images in folder {}".format(self.system_dict["dataset"]["train_path"]));
                for i in range(len(corrupt_images_train)):
                    self.custom_print("        {}. {}".format(i+1, corrupt_images_train[i]));
                self.custom_print("");
            else:
                self.custom_print("    No corrupt image found in train dir.");
                self.custom_print("");

            if(corrupt_images_val):
                self.custom_print("    Corrupt Images in folder {}".format(self.system_dict["dataset"]["val_path"]));
                for i in range(len(corrupt_images_val)):
                    self.custom_print("        {}. {}".format(i+1, corrupt_images_val[i]));
                self.custom_print("");
            else:
                self.custom_print("    No corrupt image found in val dir.");
                self.custom_print("");
    ###############################################################################################################################################



    ###############################################################################################################################################
    def Freeze_Layers(self, num=10):
        '''
        Freeze first "n" trainable layers in the network
        Args:
            num (int): Number of layers to freeze

        Returns:
            None
        '''
        self.num_freeze = num;
        self.system_dict = freeze_layers(num, self.system_dict);
        self.custom_print("Model params post freezing");
        self.custom_print("    Num trainable layers: {}".format(self.system_dict["model"]["params"]["num_params_to_update"]));
        self.custom_print("");
        save(self.system_dict);
    ###############################################################################################################################################



    ###############################################################################################################################################
    def Estimate_Train_Time(self, num_epochs=False):
        '''
        Estimate training time before running training


        Args:
            num_epochs (int): Number of epochs to be trained and get eestimation for it.

        Returns:
            None
        '''
        total_time_per_epoch = self.get_training_estimate();
        self.custom_print("Training time estimate");
        if(not num_epochs):
            total_time = total_time_per_epoch*self.system_dict["hyper-parameters"]["num_epochs"];
            self.custom_print("    {} Epochs: Approx. {} Min".format(self.system_dict["hyper-parameters"]["num_epochs"], int(total_time//60)+1));
            self.custom_print("");
        else:
            total_time = total_time_per_epoch*num_epochs;
            self.custom_print("    {} Epochs: Approx. {} Min".format(num_epochs, int(total_time//60)+1));
            self.custom_print("");
    ###############################################################################################################################################


    ##########################################################################################################################################################
    def Reload(self):
        '''
        Function to actuate all the updates in the update and expert modes


        Args:
            None

        Returns:
            None
        '''
        if(self.system_dict["states"]["eval_infer"]):
            del self.system_dict["local"]["data_loaders"];
            self.system_dict["local"]["data_loaders"] = {};
            self.Dataset();
            del self.system_dict["local"]["model"];
            self.system_dict["local"]["model"] = False;
            self.Model();
        else:
            if(not self.system_dict["states"]["copy_from"]):
                self.system_dict["local"]["model"].collect_params().reset_ctx([mx.cpu()]);
                del self.system_dict["local"]["model"];
                self.system_dict["local"]["model"] = False;
            del self.system_dict["local"]["data_loaders"];
            self.system_dict["local"]["data_loaders"] = {};
            self.Dataset();
            if(not self.system_dict["states"]["copy_from"]):
                self.Model();
                self.system_dict = load_scheduler(self.system_dict);
                self.system_dict = load_optimizer(self.system_dict);
                self.system_dict = load_loss(self.system_dict);
            if(self.system_dict["model"]["params"]["num_freeze"]):
                self.system_dict = freeze_layers(self.system_dict["model"]["params"]["num_freeze"], self.system_dict);
                self.custom_print("Model params post freezing");
                self.custom_print("    Num trainable layers: {}".format(self.system_dict["model"]["params"]["num_params_to_update"]));
                self.custom_print("");
                save(self.system_dict);
    ##########################################################################################################################################################

    
    
    
    


    ##########################################################################################################################################################
    def reset_transforms(self, test=False):
        '''
        Reset transforms to change them.


        Args:
            test (bool): If True, test transforms are reset,
                          Else, train and validation transforms are reset.

        Returns:
            None
        '''
        if(self.system_dict["states"]["eval_infer"] or test):
            self.system_dict["local"]["transforms_test"] = [];
            self.system_dict["local"]["normalize"] = False;
            self.system_dict["dataset"]["transforms"]["test"] = [];
        else:
            self.system_dict["local"]["transforms_train"] = [];
            self.system_dict["local"]["transforms_val"] = [];
            self.system_dict["local"]["normalize"] = False;
            self.system_dict["dataset"]["transforms"]["train"] = [];
            self.system_dict["dataset"]["transforms"]["val"] = [];
        save(self.system_dict);
    ##########################################################################################################################################################

    


    ##########################################################################################################################################################
    def reset_model(self):
        '''
        Reset model to update and reload it with custom weights.


        Args:
            None

        Returns:
            None
        '''
        if(self.system_dict["states"]["copy_from"]):
            msg = "Cannot reset model in Copy-From mode.\n";
            raise ConstraintError(msg)
        self.system_dict["model"]["custom_network"] = [];
        self.system_dict["model"]["final_layer"] = None;
    ##########################################################################################################################################################



    ##########################################################################################################################################################
    def Switch_Mode(self, train=False, eval_infer=False):
        '''
        Switch modes between training an inference without reloading the experiment


        Args:
            train (bool): If True, switches to training mode
            eval_infer (bool): If True, switches to validation and inferencing mode

        Returns:
            None
        '''
        if(eval_infer):
            self.system_dict["states"]["eval_infer"] = True;
        elif(train):
            self.system_dict["states"]["eval_infer"] = False;
    ##########################################################################################################################################################




    ##########################################################################################################################################################
    def debug_custom_model_design(self, network_list):
        '''
        Debug model while creating it. 
        Saves image as graph.png which is displayed 


        Args:
            network_list (list): List containing network design

        Returns:
            None
        '''
        debug_create_network(network_list);
        if(not isnotebook()):
            self.custom_print("If not using notebooks check file generated graph.png");


    ##########################################################################################################################################################


    ##########################################################################################################################################################
    def Visualize_Kernels(self, store_images_if_notebook=False):
        '''
        Visualize kernel weights of model

        Args:
            store_images_if_notebook (bool): If the images need to be stored instead of IPython widget 
                                             while using notebook. Not applicable for other environments.
        Returns:
            IPython widget displaying kernel weights if used inside a notebook.
            Else stores the maps in the visualization directory. 
        '''
        is_notebook = isnotebook()

        visualizer = CNNVisualizer(self.system_dict["local"]["model"], is_notebook)
        
        if(not is_notebook) or (store_images_if_notebook):
            self.custom_print("The images will be stored in the visualization directory of the experiment");
            
            from system.common import create_dir
            create_dir(self.system_dict["visualization"]["base"])
            create_dir(self.system_dict["visualization"]["kernels_dir"])

            visualizer.visualize_kernels(self.system_dict["visualization"]["kernels_dir"])

        else:
            visualizer.visualize_kernels()
    ##########################################################################################################################################################


    ##########################################################################################################################################################
    def Visualize_Feature_Maps(self, image_path, store_images_if_notebook=False):
        '''
        Visualize feature maps generated by model on an image

        Args:
            image_path (str): Path to the image
            store_images_if_notebook (bool): If the images need to be stored instead of IPython widget 
                                             while using notebook. Not applicable for other environments.
        
        Returns:
            IPython widget displaying feature maps if used inside a notebook.
            Else stores the maps in the visualization directory.  
        '''
        is_notebook = isnotebook()

        visualizer = CNNVisualizer(self.system_dict["local"]["model"], is_notebook)

        if(self.system_dict["model"]["params"]["use_gpu"]):
            ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
        else:
            ctx = mx.cpu()

        if(not is_notebook) or (store_images_if_notebook):
            self.custom_print("The images will be stored in the visualization directory of the experiment");

            from system.common import create_dir
            create_dir(self.system_dict["visualization"]["base"])
            create_dir(self.system_dict["visualization"]["feature_maps_dir"])
            img_name = "".join(image_path.split("/")[-1].split(".")[0:-1])
            img_dir = self.system_dict["visualization"]["feature_maps_dir"] + img_name + '/'
            create_dir(img_dir)
            
            visualizer.visualize_feature_maps(image_path, ctx=ctx, store_path=img_dir)

        else:
            visualizer.visualize_feature_maps(image_path, ctx=ctx)