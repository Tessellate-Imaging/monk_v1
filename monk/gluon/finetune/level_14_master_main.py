from gluon.finetune.imports import *
from system.imports import *

from gluon.finetune.level_13_updates_main import prototype_updates



class prototype_master(prototype_updates):
    '''
    Main class for all functions in expert mode

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
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def Dataset(self):
        '''
        Load transforms and set dataloader

        Args:
            None

        Returns:
            None
        '''
        self.set_dataset_final(test=self.system_dict["states"]["eval_infer"]);
        save(self.system_dict);

        
        if(self.system_dict["states"]["eval_infer"]):
            
            self.custom_print("Pre-Composed Test Transforms");
            self.custom_print(self.system_dict["dataset"]["transforms"]["test"]);
            self.custom_print("");

            self.custom_print("Dataset Numbers");
            self.custom_print("    Num test images: {}".format(self.system_dict["dataset"]["params"]["num_test_images"]));
            self.custom_print("    Num classes:      {}".format(self.system_dict["dataset"]["params"]["num_classes"]))
            self.custom_print("");

        else:
            
            self.custom_print("Pre-Composed Train Transforms");
            self.custom_print(self.system_dict["dataset"]["transforms"]["train"]);
            self.custom_print("");
            self.custom_print("Pre-Composed Val Transforms");
            self.custom_print(self.system_dict["dataset"]["transforms"]["val"]);
            self.custom_print("");

            self.custom_print("Dataset Numbers");
            self.custom_print("    Num train images: {}".format(self.system_dict["dataset"]["params"]["num_train_images"]));
            self.custom_print("    Num val images:   {}".format(self.system_dict["dataset"]["params"]["num_val_images"]));
            self.custom_print("    Num classes:      {}".format(self.system_dict["dataset"]["params"]["num_classes"]))
            self.custom_print("");
        
    ###############################################################################################################################################


    ###############################################################################################################################################
    @accepts("self", [int, float], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def Dataset_Percent(self, percent):
        '''
        Select a portion of dataset

        Args:
            percent (bool): percentage of sub-dataset

        Returns:
            None
        '''
        sampled_dataset = None;
        image_datasets = {};
        dataset_type = self.system_dict["dataset"]["dataset_type"];
        dataset_train_path = self.system_dict["dataset"]["train_path"];
        dataset_val_path = self.system_dict["dataset"]["val_path"];
        csv_train = self.system_dict["dataset"]["csv_train"];
        csv_val = self.system_dict["dataset"]["csv_val"];
        train_val_split = self.system_dict["dataset"]["params"]["train_val_split"];
        delimiter = self.system_dict["dataset"]["params"]["delimiter"];
        batch_size = self.system_dict["dataset"]["params"]["batch_size"];
        shuffle = self.system_dict["dataset"]["params"]["train_shuffle"];
        num_workers = self.system_dict["dataset"]["params"]["num_workers"];

        
        if(dataset_type == "train"):
            label_list = [];
            image_list = [];
            classes = os.listdir(dataset_train_path);
            for i in range(len(classes)):
                tmp_image_list = os.listdir(dataset_train_path + "/" + classes[i]);
                subset_image_list = tmp_image_list[:int(len(tmp_image_list)*percent/100.0)];
                result = list(map(lambda x: classes[i] + "/" + x, subset_image_list))
                tmp_label_list = [classes[i]]*len(subset_image_list);
                label_list += tmp_label_list;
                image_list += result;
            image_label_dict = {'ID': image_list, 'Label': label_list}  
            df = pd.DataFrame(image_label_dict);
            df.to_csv("sampled_dataset_train.csv", index=False);
        elif(dataset_type == "train-val"):
            label_list = [];
            image_list = [];
            classes = os.listdir(dataset_train_path);
            for i in range(len(classes)):
                tmp_image_list = os.listdir(dataset_train_path + "/" + classes[i]);
                subset_image_list = tmp_image_list[:int(len(tmp_image_list)*percent/100.0)];
                result = list(map(lambda x: classes[i] + "/" + x, subset_image_list))
                tmp_label_list = [classes[i]]*len(subset_image_list);
                label_list += tmp_label_list;
                image_list += result;
            image_label_dict = {'ID': image_list, 'Label': label_list}  
            df = pd.DataFrame(image_label_dict);
            df.to_csv("sampled_dataset_train.csv", index=False);

            label_list = [];
            image_list = [];
            classes = os.listdir(dataset_train_path);
            for i in range(len(classes)):
                tmp_image_list = os.listdir(dataset_val_path + "/" + classes[i]);
                subset_image_list = tmp_image_list[:int(len(tmp_image_list)*percent/100.0)];
                result = list(map(lambda x: classes[i] + "/" + x, subset_image_list))
                tmp_label_list = [classes[i]]*len(subset_image_list);
                label_list += tmp_label_list;
                image_list += result;
            image_label_dict = {'ID': image_list, 'Label': label_list}  
            df = pd.DataFrame(image_label_dict);
            df.to_csv("sampled_dataset_val.csv", index=False);
        elif(dataset_type == "csv_train"):
            df = pd.read_csv(csv_train);
            df = df.iloc[np.random.permutation(len(df))]
            df_sampled = df.iloc[:int(len(df)*percent/100.0)];
            df_sampled.to_csv("sampled_dataset_train.csv", index=False);
        elif(dataset_type == "csv_train-val"):
            df = pd.read_csv(csv_train);
            df = df.iloc[np.random.permutation(len(df))]
            df_sampled = df.iloc[:int(len(df)*percent/100.0)];
            df_sampled.to_csv("sampled_dataset_train.csv", index=False);
            df = pd.read_csv(csv_val);
            df = df.iloc[np.random.permutation(len(df))]
            df_sampled = df.iloc[:int(len(df)*percent/100.0)];
            df_sampled.to_csv("sampled_dataset_val.csv", index=False);



    ###############################################################################################################################################





    ###############################################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def Model(self):
        '''
        Load Model as per paraameters set

        Args:
            None

        Returns:
            None
        '''
        if(self.system_dict["states"]["copy_from"]):
            msg = "Cannot set model in Copy-From mode.\n";
            raise ConstraintError(msg)
        self.set_model_final();
        save(self.system_dict)
    ###############################################################################################################################################




    ###############################################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def Train(self):
        '''
        Master function for training

        Args:
            None

        Returns:
            None
        '''
        self.set_training_final();
        save(self.system_dict);
    ###############################################################################################################################################



    ###############################################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def Evaluate(self):
        '''
        Master function for external validation

        Args:
            None

        Returns:
            None
        '''
        if(self.system_dict["dataset"]["label_type"] == "single" or self.system_dict["dataset"]["label_type"] == False):
            accuracy, class_based_accuracy = self.set_evaluation_final();
            save(self.system_dict);
        else:
            accuracy, class_based_accuracy = self.set_evaluation_final_multiple();
            save(self.system_dict);
        return accuracy, class_based_accuracy;
    ###############################################################################################################################################


    ###############################################################################################################################################
    @error_checks(None, img_name=["file", "r"], img_dir=["folder", "r"], return_raw=False, img_thresh=["gte", 0.0, "lte", 1.0], post_trace=False)
    @accepts("self", img_name=[str, bool], img_dir=[str, bool], return_raw=bool, img_thresh=float, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def Infer(self, img_name=False, img_dir=False, return_raw=False, img_thresh=0.5):
        '''
        Master function for inference 

        Args:
            img_name (str): path to image
            img_dir (str): path to folders containing images. 
                            (Optional)
            return_raw (bool): If True, then output dictionary contains image probability for every class in the set.
                                Else, only the most probable class score is returned back.
            img_thresh (float): Thresholding for multi label image classification.

        Returns:
            dict: Dictionary containing details on predictions.
        '''
        if(self.system_dict["dataset"]["label_type"] == "single" or self.system_dict["dataset"]["label_type"] == False):
            if(not img_dir):
                predictions = self.set_prediction_final(img_name=img_name, return_raw=return_raw);
            else:
                predictions = self.set_prediction_final(img_dir=img_dir, return_raw=return_raw);
            return predictions;
        else:
            if(not img_dir):
                predictions = self.set_prediction_final_multiple(img_name=img_name, return_raw=return_raw, img_thresh=img_thresh);
            else:
                predictions = self.set_prediction_final_multiple(img_dir=img_dir, return_raw=return_raw, img_thresh=img_thresh);
            return predictions;
    ###############################################################################################################################################



    ###############################################################################################################################################
    @accepts("self", network=list, data_shape=tuple, use_gpu=bool, network_initializer=str, debug=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def Compile_Network(self, network, data_shape=(3, 224, 224), use_gpu=True, network_initializer="xavier_normal", debug=True):
        '''
        Master function for compiling custom network and initializing it 

        Args:
            network: Network stacked as list of lists
            data_shape (tuple): Input shape of data in format C, H, W
            use_gpu (bool): If True, model loaded on gpu
            network_initializer (str): Initialize network with random weights. Select the random generator type function.

        Returns:
            None
        '''
        self.system_dict["custom_model"]["network_stack"] = network;
        self.system_dict["custom_model"]["network_initializer"] = network_initializer;
        self.system_dict["model"]["type"] = "custom";
        self.system_dict["dataset"]["params"]["data_shape"] = data_shape;
        self.system_dict["custom_model"]["debug"] = debug;
        self.system_dict = set_device(use_gpu, self.system_dict);
        save(self.system_dict);
        self.set_model_final();
    ###############################################################################################################################################


    ###############################################################################################################################################
    @accepts("self", data_shape=tuple, port=int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def Visualize_With_Netron(self, data_shape=None, port=None):
        '''
        Visualize network with netron library 

        Args:
            data_shape (tuple): Input shape of data in format C, H, W
            port (int): Local host free port.

        Returns:
            None
        '''
        self.custom_print("Using Netron To Visualize");
        self.custom_print("Not compatible on kaggle");
        self.custom_print("Compatible only for Jupyter Notebooks");

        if not data_shape:
            c, h, w = self.system_dict["dataset"]["params"]["data_shape"];
        else:
            c, h, w = data_shape;

        data = mx.nd.random.randn(1, c, h, w)
        if(self.system_dict["model"]["params"]["use_gpu"]):
            self.system_dict["local"]["ctx"] = [mx.gpu(0)];
        else:
            self.system_dict["local"]["ctx"] = [mx.cpu()];

        data = data.copyto(self.system_dict["local"]["ctx"][0])

        self.system_dict["local"]["model"].hybridize();
        out = self.system_dict["local"]["model"](data);

        self.system_dict["local"]["model"].export("model", epoch=0)

        import netron
        if(not port):
            netron.start('model-symbol.json')
        else:
            netron.start('model-symbol.json', port=port)
    ###############################################################################################################################################