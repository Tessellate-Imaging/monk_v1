from tf_keras.finetune.imports import *
from system.imports import *

from tf_keras.finetune.level_14_master_main import prototype_master



class prototype(prototype_master):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);
        self.system_dict["library"] = "Keras";
        self.custom_print("Keras Version: {}".format(keras.__version__));
        self.custom_print("Tensorflow Version: {}".format(tf.__version__));
        self.custom_print("");


    ###############################################################################################################################################
    @error_checks(None, ["name", ["A-Z", "a-z", "0-9", "-", "_"]], ["name", ["A-Z", "a-z", "0-9", "-", "_"]], 
        eval_infer=None, resume_train=None, copy_from=None, summary=None, post_trace=True)
    @accepts("self", str, str, eval_infer=bool, resume_train=bool, copy_from=[list, bool], summary=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Prototype(self, project_name, experiment_name, eval_infer=False, resume_train=False, copy_from=False, summary=False):
        self.set_system_project(project_name);
        self.set_system_experiment(experiment_name, eval_infer=eval_infer, resume_train=resume_train, copy_from=copy_from, summary=summary);
        self.custom_print("Experiment Details");
        self.custom_print("    Project: {}".format(self.system_dict["project_name"]));
        self.custom_print("    Experiment: {}".format(self.system_dict["experiment_name"]));
        self.custom_print("    Dir: {}".format(self.system_dict["experiment_dir"]));
        self.custom_print("");
    ###############################################################################################################################################




    ###############################################################################################################################################
    @warning_checks(None, dataset_path=None, path_to_csv=None, delimiter=None,
        model_name=None, freeze_base_network=None, num_epochs=["lt", 100], post_trace=True)
    @error_checks(None, dataset_path=["folder", "r"], path_to_csv=["file", "r"], delimiter=["in", [",", ";", "-", " "]],
        model_name=None, freeze_base_network=None, num_epochs=["gte", 1], post_trace=True)
    @accepts("self", dataset_path=[str, list, bool], path_to_csv=[str, list, bool], delimiter=str, 
        model_name=str, freeze_base_network=bool, num_epochs=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Default(self, dataset_path=False, path_to_csv=False, delimiter=",", model_name="resnet18_v1", freeze_base_network=True, num_epochs=10):
        if(self.system_dict["states"]["eval_infer"]):
            self.Dataset_Params(dataset_path=dataset_path, import_as_csv=import_as_csv, path_to_csv=path_to_csv, delimiter=delimiter);
            self.Dataset();
        else:
            input_size=224;
            self.Dataset_Params(dataset_path=dataset_path, path_to_csv=path_to_csv, delimiter=delimiter, 
                split=0.7, input_size=input_size, batch_size=4, shuffle_data=True, num_processors=psutil.cpu_count());

            self.apply_random_horizontal_flip(probability=0.8, train=True, val=True);
            self.apply_mean_subtraction(mean=[0.485, 0.456, 0.406], train=True, val=True, test=True);
            self.Dataset();

            self.Model_Params(model_name=model_name, freeze_base_network=freeze_base_network, use_gpu=True, gpu_memory_fraction=0.6, use_pretrained=True);
            self.Model();

            model_name = self.system_dict["model"]["params"]["model_name"];


            if("resnet" in model_name or "vgg" in model_name or "dense" in model_name or "xception" in model_name):
                self.optimizer_sgd(0.0001, momentum=0.9);
                #self.lr_plateau_decrease(factor=0.1, patience=max(min(10, num_epochs//3), 1), verbose=True);
                self.lr_step_decrease(1, gamma=0.97); #Testing for Issue - Issue with Keras training #5
                self.loss_crossentropy();
            elif("nas" in model_name):
                self.optimizer_rmsprop(0.0001, weight_decay=0.00004, momentum=0.9);
                self.lr_step_decrease(2, gamma=0.97);
                self.loss_crossentropy();
            elif("mobile" in model_name):
                self.optimizer_sgd(0.0001, weight_decay=0.00004, momentum=0.9);
                self.lr_step_decrease(1, gamma=0.97);
                self.loss_crossentropy();
            elif("inception" in model_name):
                self.optimizer_sgd(0.0001, weight_decay=0.0001, momentum=0.9);
                self.lr_step_decrease(1, gamma=0.9);
                self.loss_crossentropy();

            self.Training_Params(num_epochs=num_epochs, display_progress=True, display_progress_realtime=True, 
            save_intermediate_models=True, intermediate_model_prefix="intermediate_model_", save_training_logs=True);

            self.system_dict["hyper-parameters"]["status"] = True;

            save(self.system_dict);
    ###############################################################################################################################################




    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Summary(self):
        print_summary(self.system_dict["fname"]);
    ###############################################################################################################################################


    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def List_Models(self):
        self.print_list_models();
    ###############################################################################################################################################






    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def List_Layers(self):
        self.print_list_layers();
    ###############################################################################################################################################





    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def List_Activations(self):
        self.print_list_activations();
    ###############################################################################################################################################







    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def List_Losses(self):
        self.print_list_losses();
    ###############################################################################################################################################







    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def List_Optimizers(self):
        self.print_list_optimizers();
    ###############################################################################################################################################







    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def List_Schedulers(self):
        self.print_list_schedulers();
    ###############################################################################################################################################





    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def List_Transforms(self):
        self.print_list_transforms();
    ###############################################################################################################################################