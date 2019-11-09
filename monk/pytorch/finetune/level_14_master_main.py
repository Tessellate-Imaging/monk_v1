from pytorch.finetune.imports import *
from system.imports import *

from pytorch.finetune.level_13_updates_main import prototype_updates



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