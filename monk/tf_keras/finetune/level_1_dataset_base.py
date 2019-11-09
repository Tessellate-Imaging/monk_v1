from system.imports import *
from tf_keras.finetune.imports import *

from system.base_class import system


class finetune_dataset(system):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    ###############################################################################################################################################
    @accepts("self", test=bool, estimate=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def set_dataset_dataloader(self, test=False, estimate=False):
        if(test):
            num_workers = self.system_dict["dataset"]["params"]["num_workers"];
            csv_test = self.system_dict["dataset"]["csv_test"];
            dataset_test_path = self.system_dict["dataset"]["test_path"];
            delimiter = self.system_dict["dataset"]["params"]["test_delimiter"];

            target_size = (self.system_dict["dataset"]["params"]["input_size"], self.system_dict["dataset"]["params"]["input_size"]);
            color_mode='rgb';
            class_mode='categorical';

            if(self.system_dict["dataset"]["params"]["dataset_test_type"] == "foldered"):
                self.system_dict["local"]["data_loaders"]["test"] = self.system_dict["local"]["data_generators"]["test"].flow_from_directory(dataset_test_path, 
                                                                 target_size=target_size,
                                                                 color_mode=color_mode,
                                                                 batch_size=1,
                                                                 class_mode=class_mode,
                                                                 shuffle=False
                                                                 );

            elif(self.system_dict["dataset"]["params"]["dataset_test_type"] == "csv"):
                test_df, columns = parse_csv2(csv_test, delimiter);
                self.system_dict["local"]["data_loaders"]["test"] = self.system_dict["local"]["data_generators"]["test"].flow_from_dataframe(
                                                                dataframe=test_df,
                                                                directory=dataset_test_path,
                                                                x_col=columns[0],
                                                                y_col=columns[1],
                                                                target_size=target_size,
                                                                color_mode=color_mode,
                                                                batch_size=1,
                                                                class_mode=class_mode,
                                                                shuffle=False
                                                                );

            if(not self.system_dict["dataset"]["params"]["classes"]):
                self.system_dict["dataset"]["params"]["classes"] = self.system_dict["local"]["data_loaders"]["test"].class_indices;

            self.system_dict["dataset"]["params"]["num_test_images"] = len(self.system_dict["local"]["data_loaders"]["test"].labels);


        elif(estimate):
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

            target_size = (self.system_dict["dataset"]["params"]["input_size"], self.system_dict["dataset"]["params"]["input_size"]);
            color_mode='rgb';
            class_mode='categorical';


            if("csv" in dataset_type):
                train_df, columns = parse_csv2(csv_train, delimiter);
                self.system_dict["local"]["data_loaders"]["estimate"] = self.system_dict["local"]["data_generators"]["estimate"].flow_from_dataframe(
                                                                dataframe=train_df,
                                                                directory=dataset_train_path,
                                                                x_col=columns[0],
                                                                y_col=columns[1],
                                                                subset="training",
                                                                target_size=target_size,
                                                                color_mode=color_mode,
                                                                batch_size=batch_size,
                                                                class_mode=class_mode,
                                                                shuffle=shuffle
                                                            );
            else:
                self.system_dict["local"]["data_loaders"]["estimate"] = self.system_dict["local"]["data_generators"]["estimate"].flow_from_directory(dataset_train_path, 
                                                                 target_size=target_size,
                                                                 color_mode=color_mode,
                                                                 batch_size=batch_size,
                                                                 class_mode=class_mode,
                                                                 shuffle=shuffle,
                                                                 subset='training');


        else:
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


            target_size = (self.system_dict["dataset"]["params"]["input_size"], self.system_dict["dataset"]["params"]["input_size"]);
            color_mode='rgb';
            class_mode='categorical';

            if(dataset_type == "train"):
                self.system_dict["local"]["data_loaders"]["train"] = self.system_dict["local"]["data_generators"]["train"].flow_from_directory(dataset_train_path, 
                                                                 target_size=target_size,
                                                                 color_mode=color_mode,
                                                                 batch_size=batch_size,
                                                                 class_mode=class_mode,
                                                                 shuffle=shuffle,
                                                                 subset='training');

                self.system_dict["local"]["data_loaders"]["val"] = self.system_dict["local"]["data_generators"]["train"].flow_from_directory(dataset_train_path, 
                                                                 target_size=target_size,
                                                                 color_mode=color_mode,
                                                                 batch_size=batch_size,
                                                                 class_mode=class_mode,
                                                                 shuffle=shuffle,
                                                                 subset='validation');

            elif(dataset_type == "train-val"):
                self.system_dict["local"]["data_loaders"]["train"] = self.system_dict["local"]["data_generators"]["train"].flow_from_directory(dataset_train_path, 
                                                                 target_size=target_size,
                                                                 color_mode=color_mode,
                                                                 batch_size=batch_size,
                                                                 class_mode=class_mode,
                                                                 shuffle=shuffle);

                self.system_dict["local"]["data_loaders"]["val"] = self.system_dict["local"]["data_generators"]["val"].flow_from_directory(dataset_val_path, 
                                                                 target_size=target_size,
                                                                 color_mode=color_mode,
                                                                 batch_size=batch_size,
                                                                 class_mode=class_mode,
                                                                 shuffle=shuffle);


            elif(dataset_type == "csv_train"):
                train_df, columns = parse_csv2(csv_train, delimiter);

                self.system_dict["local"]["data_loaders"]["train"] = self.system_dict["local"]["data_generators"]["train"].flow_from_dataframe(
                                                                dataframe=train_df,
                                                                directory=dataset_train_path,
                                                                x_col=columns[0],
                                                                y_col=columns[1],
                                                                subset="training",
                                                                target_size=target_size,
                                                                color_mode=color_mode,
                                                                batch_size=batch_size,
                                                                class_mode=class_mode,
                                                                shuffle=shuffle
                                                            );

                self.system_dict["local"]["data_loaders"]["val"] = self.system_dict["local"]["data_generators"]["train"].flow_from_dataframe(
                                                                dataframe=train_df,
                                                                directory=dataset_train_path,
                                                                x_col=columns[0],
                                                                y_col=columns[1],
                                                                subset="validation",
                                                                target_size=target_size,
                                                                color_mode=color_mode,
                                                                batch_size=batch_size,
                                                                class_mode=class_mode,
                                                                shuffle=shuffle
                                                            );


            elif(dataset_type == "csv_train-val"):
                train_df, columns = parse_csv2(csv_train, delimiter);


                self.system_dict["local"]["data_loaders"]["train"] = self.system_dict["local"]["data_generators"]["train"].flow_from_dataframe(
                                                                dataframe=train_df,
                                                                directory=dataset_train_path,
                                                                x_col=columns[0],
                                                                y_col=columns[1],
                                                                target_size=target_size,
                                                                color_mode=color_mode,
                                                                batch_size=batch_size,
                                                                class_mode=class_mode,
                                                                shuffle=shuffle
                                                            );


                val_df, columns = parse_csv2(csv_val, delimiter);

                self.system_dict["local"]["data_loaders"]["val"] = self.system_dict["local"]["data_generators"]["val"].flow_from_dataframe(
                                                                dataframe=val_df,
                                                                directory=dataset_val_path,
                                                                x_col=columns[0],
                                                                y_col=columns[1],
                                                                target_size=target_size,
                                                                color_mode=color_mode,
                                                                batch_size=batch_size,
                                                                class_mode=class_mode,
                                                                shuffle=shuffle
                                                            );


            self.system_dict["dataset"]["params"]["classes"] = self.system_dict["local"]["data_loaders"]["train"].class_indices;
            self.system_dict["dataset"]["params"]["num_classes"] = len(self.system_dict["dataset"]["params"]["classes"]);

            self.system_dict["dataset"]["params"]["num_train_images"] = len(self.system_dict["local"]["data_loaders"]["train"].labels);
            self.system_dict["dataset"]["params"]["num_val_images"] = len(self.system_dict["local"]["data_loaders"]["val"].labels);

            self.system_dict["dataset"]["status"]= True;
    ###############################################################################################################################################







    ###############################################################################################################################################
    @accepts("self", test=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def set_dataset_final(self, test=False):
        if(test):
            self.system_dict = set_transform_test(self.system_dict);
        else:
            self.system_dict = set_transform_trainval(self.system_dict);
        self.set_dataset_dataloader(test=test);
    ###############################################################################################################################################

    