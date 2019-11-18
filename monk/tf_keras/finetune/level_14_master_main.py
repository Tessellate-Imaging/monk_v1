from tf_keras.finetune.imports import *
from system.imports import *

from tf_keras.finetune.level_13_updates_main import prototype_updates



class prototype_master(prototype_updates):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Dataset(self):
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
    @accepts("self", [int, float], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Dataset_Percent(self, percent):
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
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Model(self):
        if(self.system_dict["states"]["copy_from"]):
            msg = "Cannot set model in Copy-From mode.\n";
            raise ConstraintError(msg)
        self.set_model_final();
        save(self.system_dict)
    ###############################################################################################################################################


    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Train(self):
        self.set_training_final();
        save(self.system_dict);
    ###############################################################################################################################################





    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Evaluate(self):
        accuracy, class_based_accuracy = self.set_evaluation_final();
        save(self.system_dict);
        return accuracy, class_based_accuracy;
    ###############################################################################################################################################




    ###############################################################################################################################################
    @error_checks(None, img_name=["file", "r"], img_dir=["folder", "r"], return_raw=False, post_trace=True)
    @accepts("self", img_name=[str, bool], img_dir=[str, bool], return_raw=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Infer(self, img_name=False, img_dir=False, return_raw=False):
        if(not img_dir):
            predictions = self.set_prediction_final(img_name=img_name, return_raw=return_raw);
        else:
            predictions = self.set_prediction_final(img_dir=img_dir, return_raw=return_raw);
        return predictions;
    ###############################################################################################################################################