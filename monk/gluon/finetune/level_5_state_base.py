from gluon.finetune.imports import *
from system.imports import *

from gluon.finetune.level_4_evaluation_base import finetune_evaluation




class finetune_state(finetune_evaluation):
    '''
    Base class for Monk states - train, eval_infer, resume, copy_from, pseudo_copy_from (for running sub-experiments)

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
    def set_system_state_eval_infer(self):
        '''
        Set system for eval_infer state

        Args:
            None

        Returns:
            None
        '''
        self.system_dict = read_json(self.system_dict["fname"], verbose=self.system_dict["verbose"]);
        self.system_dict["states"]["eval_infer"] = True;

        if(self.system_dict["training"]["status"]):
            if(len(self.system_dict["dataset"]["transforms"]["test"])):
                self.system_dict = retrieve_test_transforms(self.system_dict);
            else:
                self.custom_print("Test transforms not found.");
                self.custom_print("Add test transforms");
                self.custom_print("");
            self.set_model_final();
        else:
            msg = "Model in {} not trained. Cannot perform testing or inferencing".format(self.system_dict["experiment_name"]);
            raise ConstraintError(msg);
    ###############################################################################################################################################


    ###############################################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_system_state_resume_train(self):
        '''
        Set system for resume training state

        Args:
            None

        Returns:
            None
        '''
        self.system_dict = read_json(self.system_dict["fname"], verbose=self.system_dict["verbose"]);
        self.system_dict["states"]["resume_train"] = True;
        if(self.system_dict["dataset"]["status"]):
            self.system_dict = retrieve_trainval_transforms(self.system_dict);
            self.set_dataset_final();
        else:
            msg = "Dataset not set.\n";
            msg += "Training not started. Cannot Run resume Mode";
            raise ConstraintError(msg);
        if(self.system_dict["model"]["status"]):
            self.set_model_final();
        else:
            msg = "Model not set.\n";
            msg += "Training not started. Cannot Run resume Mode";
            raise ConstraintError(msg);
        if(self.system_dict["hyper-parameters"]["status"]):
            self.system_dict = retrieve_optimizer(self.system_dict);
            self.system_dict = retrieve_scheduler(self.system_dict);
            self.system_dict = retrieve_loss(self.system_dict);
        else:
            msg = "hyper-parameters not set.\n";
            msg += "Training not started. Cannot Run resume Mode";
            raise ConstraintError(msg);
    ###############################################################################################################################################


    ###############################################################################################################################################
    @accepts("self", list, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_system_state_copy_from(self, copy_from):
        '''
        Set system for copied state

        Args:
            None

        Returns:
            None
        '''
        fname = self.system_dict["master_systems_dir_relative"] + copy_from[0] + "/" + copy_from[1] + "/experiment_state.json";
        system_dict_tmp = read_json(fname, verbose=self.system_dict["verbose"]);

        if(not system_dict_tmp["training"]["status"]):
            self.custom_print("Project - {}, Experiment - {}, has incomplete training.".format(copy_from[0], copy_from[1]));
            self.custom_print("Complete Previous training before copying from it.");
            self.custom_print("");
        elif(copy_from[0] == self.system_dict["project_name"] and copy_from[1] == self.system_dict["experiment_name"]):
            self.custom_print("Cannot copy same experiment. Use a different experiment to copy and load a previous experiment");
            self.custom_print("");
        else:
            self.system_dict["dataset"] = system_dict_tmp["dataset"];
            self.system_dict["model"] = system_dict_tmp["model"];
            self.system_dict["hyper-parameters"] = system_dict_tmp["hyper-parameters"];
            self.system_dict["training"] = system_dict_tmp["training"];
            self.system_dict["origin"] = [copy_from[0], copy_from[1]];
            self.system_dict["training"]["outputs"] = {};
            self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] = 0;
            self.system_dict["training"]["outputs"]["best_val_acc"] = 0;
            self.system_dict["training"]["outputs"]["best_val_acc_epoch_num"] = 0;
            self.system_dict["training"]["outputs"]["epochs_completed"] = 0;
            self.system_dict["training"]["status"] = False;
            self.system_dict["training"]["enabled"] = True;
            self.system_dict["testing"] = {};
            self.system_dict["testing"]["status"] = False;
            save(self.system_dict);

            self.system_dict = read_json(self.system_dict["fname_relative"], verbose=self.system_dict["verbose"]);
            self.system_dict["states"]["copy_from"] = True;
            self.system_dict = retrieve_trainval_transforms(self.system_dict);
            self.Dataset();
            self.set_model_final();
            self.system_dict = retrieve_optimizer(self.system_dict);
            self.system_dict = retrieve_scheduler(self.system_dict);
            self.system_dict = retrieve_loss(self.system_dict);
    ###############################################################################################################################################



    ###############################################################################################################################################
    @accepts("self", list, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_system_state_pseudo_copy_from(self, pseudo_copy_from):
        '''
        Set system for copied state in hyper-parameter analysis mode 

        Args:
            None

        Returns:
            None
        '''
        fname = self.system_dict["master_systems_dir_relative"] + pseudo_copy_from[0] + "/" + pseudo_copy_from[1] + "/experiment_state.json";
        system_dict_tmp = read_json(fname, verbose=self.system_dict["verbose"]);
        self.system_dict["dataset"] = system_dict_tmp["dataset"];
        self.system_dict["model"] = system_dict_tmp["model"];
        self.system_dict["hyper-parameters"] = system_dict_tmp["hyper-parameters"];
        self.system_dict["training"] = system_dict_tmp["training"];
        self.system_dict["origin"] = [pseudo_copy_from[0], pseudo_copy_from[1]];
        self.system_dict["training"]["outputs"] = {};
        self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] = 0;
        self.system_dict["training"]["outputs"]["best_val_acc"] = 0;
        self.system_dict["training"]["outputs"]["best_val_acc_epoch_num"] = 0;
        self.system_dict["training"]["outputs"]["epochs_completed"] = 0;
        self.system_dict["training"]["status"] = False;
        self.system_dict["training"]["enabled"] = True;
        self.system_dict["testing"] = {};
        self.system_dict["testing"]["status"] = False;
        save(self.system_dict);

        self.system_dict = read_json(self.system_dict["fname_relative"], verbose=self.system_dict["verbose"]);
        self.system_dict["states"]["pseudo_copy_from"] = True;
        self.system_dict = retrieve_trainval_transforms(self.system_dict);
        self.Dataset();
        self.set_model_final();
        self.system_dict = retrieve_optimizer(self.system_dict);
        self.system_dict = retrieve_scheduler(self.system_dict);
        self.system_dict = retrieve_loss(self.system_dict);
    ###############################################################################################################################################

    