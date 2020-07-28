from monk.system.imports import *
from monk.gluon.finetune.imports import *

from monk.system.base_class import system


class finetune_dataset(system):
    '''
    Base class for dataset params

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
    @accepts("self", test=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_dataset_dataloader(self, test=False):
        '''
        Setup the dataloader.

        Args:
            test (bool): If True then test data is loaded, else train data is loaded.

        Returns:
            None
        '''
        if(test):
            label_type = self.system_dict["dataset"]["label_type"];
            
            if(label_type == "single" or label_type == False):
                num_workers = self.system_dict["dataset"]["params"]["num_workers"];
                if(self.system_dict["dataset"]["params"]["dataset_test_type"] == "foldered"):
                    test_dataset = mx.gluon.data.vision.ImageFolderDataset(self.system_dict["dataset"]["test_path"]).transform_first(self.system_dict["local"]["data_transforms"]["test"]);
                    if(not self.system_dict["dataset"]["params"]["classes"]):
                        self.system_dict["dataset"]["params"]["classes"] = list(np.unique(sorted(os.listdir(self.system_dict["dataset"]["test_path"]))));
                elif(self.system_dict["dataset"]["params"]["dataset_test_type"] == "csv"):
                    if(not self.system_dict["dataset"]["params"]["classes"]):
                        img_list, label_list, self.system_dict["dataset"]["params"]["classes"] = parse_csv(self.system_dict["dataset"]["csv_test"], self.system_dict["dataset"]["params"]["test_delimiter"]);
                    else:
                        img_list, label_list, _ = parse_csv(self.system_dict["dataset"]["csv_test"], self.system_dict["dataset"]["params"]["test_delimiter"]);
                    test_dataset = DatasetCustom(img_list, label_list, self.system_dict["dataset"]["test_path"]).transform_first(self.system_dict["local"]["data_transforms"]["test"]);
                self.system_dict["local"]["data_loaders"]['test'] = mx.gluon.data.DataLoader(test_dataset, batch_size=1, shuffle = False, num_workers=num_workers);
                self.system_dict["dataset"]["params"]["num_test_images"] = len(self.system_dict["local"]["data_loaders"]['test']);

            else:
                num_workers = self.system_dict["dataset"]["params"]["num_workers"];
                if(not self.system_dict["dataset"]["params"]["classes"]):
                    img_list, label_list, self.system_dict["dataset"]["params"]["classes"] = parse_csv_updated(self.system_dict["dataset"]["csv_test"], 
                                                                                                    self.system_dict["dataset"]["params"]["test_delimiter"]);
                else:
                    img_list, label_list, _ = parse_csv_updated(self.system_dict["dataset"]["csv_test"], self.system_dict["dataset"]["params"]["test_delimiter"]);

                test_dataset = DatasetCustomMultiLabel(img_list, label_list, 
                                                        self.system_dict["dataset"]["params"]["classes"], 
                                                        self.system_dict["dataset"]["test_path"]).transform_first(self.system_dict["local"]["data_transforms"]["test"]);

                self.system_dict["local"]["data_loaders"]['test'] = mx.gluon.data.DataLoader(test_dataset, batch_size=1, shuffle = False, num_workers=num_workers);
                self.system_dict["dataset"]["params"]["num_test_images"] = len(self.system_dict["local"]["data_loaders"]['test']);

        else:
            sampled_dataset = None;
            image_datasets = {};
            dataset_type = self.system_dict["dataset"]["dataset_type"];
            label_type = self.system_dict["dataset"]["label_type"];
            dataset_train_path = self.system_dict["dataset"]["train_path"];
            dataset_val_path = self.system_dict["dataset"]["val_path"];
            csv_train = self.system_dict["dataset"]["csv_train"];
            csv_val = self.system_dict["dataset"]["csv_val"];
            train_val_split = self.system_dict["dataset"]["params"]["train_val_split"];
            delimiter = self.system_dict["dataset"]["params"]["delimiter"];
            batch_size = self.system_dict["dataset"]["params"]["batch_size"];
            shuffle = self.system_dict["dataset"]["params"]["train_shuffle"];
            num_workers = self.system_dict["dataset"]["params"]["num_workers"];



            if(label_type == "single" or label_type == False):
                if(dataset_type == "train"):
                    sampled_dataset = mx.gluon.data.vision.ImageFolderDataset(dataset_train_path).transform_first(self.system_dict["local"]["data_transforms"]["train"]);
                elif(dataset_type == "train-val"):
                    image_datasets["train"] = mx.gluon.data.vision.ImageFolderDataset(dataset_train_path).transform_first(self.system_dict["local"]["data_transforms"]["train"]);
                    image_datasets["val"] = mx.gluon.data.vision.ImageFolderDataset(dataset_val_path).transform_first(self.system_dict["local"]["data_transforms"]["val"]);
                elif(dataset_type == "csv_train"):
                    img_list, label_list, self.system_dict["dataset"]["params"]["classes"] = parse_csv(csv_train, delimiter);
                    sampled_dataset = DatasetCustom(img_list, label_list, dataset_train_path).transform_first(self.system_dict["local"]["data_transforms"]["train"]);
                elif(dataset_type == "csv_train-val"):
                    img_list, label_list, self.system_dict["dataset"]["params"]["classes"] = parse_csv(csv_train, delimiter);
                    image_datasets["train"] = DatasetCustom(img_list, label_list, dataset_train_path).transform_first(self.system_dict["local"]["data_transforms"]["train"]);
                    img_list, label_list, self.system_dict["dataset"]["params"]["classes"] = parse_csv(csv_val, delimiter);
                    image_datasets["val"] = DatasetCustom(img_list, label_list, dataset_val_path).transform_first(self.system_dict["local"]["data_transforms"]["val"]);



                if(sampled_dataset):
                    lengths = [int(len(sampled_dataset)*train_val_split), len(sampled_dataset) - int(len(sampled_dataset)*train_val_split)]
                    image_datasets["train"], image_datasets["val"] = torch.utils.data.dataset.random_split(sampled_dataset, lengths)


                if("csv" not in dataset_type):
                    self.system_dict["dataset"]["params"]["classes"] = list(np.unique(sorted(os.listdir(dataset_train_path))));

                self.system_dict["dataset"]["params"]["num_classes"] = len(self.system_dict["dataset"]["params"]["classes"]);

                
                if(self.system_dict["dataset"]["params"]["weighted_sample"]):
                    self.custom_print("    Weighted sampling enabled.");
                    self.custom_print("    Weighted sampling temporarily suspended. Skipping step");


                self.system_dict["dataset"]["params"]["num_train_images"] = len(image_datasets["train"]);
                self.system_dict["dataset"]["params"]["num_val_images"] = len(image_datasets["val"]);


                if(self.system_dict["dataset"]["params"]["weighted_sample"]):
                    self.system_dict["local"]["data_loaders"]["train"] = mx.gluon.data.DataLoader(image_datasets["train"],
                                                                    batch_size=batch_size, shuffle=True, num_workers=num_workers);
                else:
                    self.system_dict["local"]["data_loaders"]["train"] = mx.gluon.data.DataLoader(image_datasets["train"],
                                                                    batch_size=batch_size, shuffle=shuffle, num_workers=num_workers);

                self.system_dict["local"]["data_loaders"]["val"] = mx.gluon.data.DataLoader(image_datasets["val"],
                                                                    batch_size=batch_size, shuffle=shuffle, num_workers=num_workers);

                

                self.system_dict["dataset"]["status"]= True;

            else:

                if(dataset_type == "csv_train"):
                    img_list, label_list, self.system_dict["dataset"]["params"]["classes"] = parse_csv_updated(csv_train, delimiter);
                    sampled_dataset = DatasetCustomMultiLabel(img_list, label_list, 
                                                                self.system_dict["dataset"]["params"]["classes"], 
                                                                dataset_train_path).transform_first(self.system_dict["local"]["data_transforms"]["train"]);
                elif(dataset_type == "csv_train-val"):
                    img_list, label_list, self.system_dict["dataset"]["params"]["classes"] = parse_csv_updated(csv_train, delimiter);
                    image_datasets["train"] = DatasetCustomMultiLabel(img_list, label_list, 
                                                                self.system_dict["dataset"]["params"]["classes"], 
                                                                dataset_train_path).transform_first(self.system_dict["local"]["data_transforms"]["train"]);
                    img_list, label_list, _ = parse_csv_updated(csv_val, delimiter);
                    image_datasets["val"] = DDatasetCustomMultiLabel(img_list, label_list, 
                                                                self.system_dict["dataset"]["params"]["classes"], 
                                                                dataset_val_path).transform_first(self.system_dict["local"]["data_transforms"]["val"]);

                if(sampled_dataset):
                    lengths = [int(len(sampled_dataset)*train_val_split), len(sampled_dataset) - int(len(sampled_dataset)*train_val_split)]
                    image_datasets["train"], image_datasets["val"] = torch.utils.data.dataset.random_split(sampled_dataset, lengths)

                self.system_dict["dataset"]["params"]["num_classes"] = len(self.system_dict["dataset"]["params"]["classes"]);

                self.system_dict["dataset"]["params"]["num_train_images"] = len(image_datasets["train"]);
                self.system_dict["dataset"]["params"]["num_val_images"] = len(image_datasets["val"]);


                if(self.system_dict["dataset"]["params"]["weighted_sample"]):
                    self.system_dict["local"]["data_loaders"]["train"] = mx.gluon.data.DataLoader(image_datasets["train"],
                                                                    batch_size=batch_size, shuffle=True, num_workers=num_workers);
                else:
                    self.system_dict["local"]["data_loaders"]["train"] = mx.gluon.data.DataLoader(image_datasets["train"],
                                                                    batch_size=batch_size, shuffle=shuffle, num_workers=num_workers);

                self.system_dict["local"]["data_loaders"]["val"] = mx.gluon.data.DataLoader(image_datasets["val"],
                                                                    batch_size=batch_size, shuffle=shuffle, num_workers=num_workers);

                

                self.system_dict["dataset"]["status"]= True;



    ###############################################################################################################################################


    ###############################################################################################################################################
    @accepts("self", test=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_dataset_final(self, test=False):
        '''
        Set the transforms and then invoke data loader.

        Args:
            test (bool): If True then test tranforms are set and test dataloader is prepared data, 
                        else train transforms are set and train dataloader is prepared.

        Returns:
            None
        '''
        if(test):
            self.system_dict = set_transform_test(self.system_dict);
        else:
            self.system_dict = set_transform_trainval(self.system_dict);
        self.set_dataset_dataloader(test=test);
    ###############################################################################################################################################