from gluon.finetune.imports import *
from system.imports import *
from gluon.finetune.level_1_dataset_base import finetune_dataset





class finetune_model(finetune_dataset):
    '''
    Base class for Model setup

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
    @accepts("self", path=[bool, str, list], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_model_final(self, path=False):
        '''
        Setup model based on set parameters

        Args:
            path (str): Dummy variable

        Returns:
            None
        '''
        self.custom_print("Model Details");
        if(self.system_dict["model"]["params"]["model_path"]):
            if(os.path.isfile(self.system_dict["model"]["params"]["model_path"][0])):
                self.custom_print("    Loading model - {}".format(self.system_dict["model"]["params"]["model_path"]));
                self.system_dict["local"]["model"] = load_model(self.system_dict, external_path=self.system_dict["model"]["params"]["model_path"]);
                self.system_dict = model_to_device(self.system_dict);
                self.custom_print("    Model loaded!");
                self.custom_print("");
            else:
                msg = "Model not found - {}\n".format(self.system_dict["model"]["params"]["model_path"]);
                msg += "Previous Training Incomplete.";
                raise ConstraintError(msg);


        elif(self.system_dict["states"]["copy_from"]):
            model_path = self.system_dict["master_systems_dir_relative"] + self.system_dict["origin"][0] + "/" + self.system_dict["origin"][1] + "/output/models/";
            if(os.path.isfile(model_path + 'final-symbol.json')):
                self.custom_print("    Loading model - {}".format(model_path + 'final-symbol.json'));
                self.system_dict["local"]["model"] = load_model(self.system_dict, path=model_path, final=True);
                self.system_dict = model_to_device(self.system_dict);
                self.custom_print("    Model loaded!");
                self.custom_print("");
            else:
                msg = "Model not found - {}\n".format(model_path);
                msg += "Previous Training Incomplete.";
                raise ConstraintError(msg);

        elif(self.system_dict["states"]["eval_infer"]):
            if(os.path.isfile(self.system_dict["model_dir_relative"] + 'final-symbol.json')):
                self.custom_print("    Loading model - {}".format(self.system_dict["model_dir_relative"] + 'final-symbol.json'));
                self.system_dict["local"]["model"] = load_model(self.system_dict, final=True);
                self.system_dict = model_to_device(self.system_dict);
                self.custom_print("    Model loaded!");
                self.custom_print("");
            else:
                msg = "Model not found - {}\n".format(self.system_dict["model_dir_relative"] + 'final-symbol.json');
                msg += "Previous Training Incomplete.";
                raise ConstraintError(msg);
        else:
            if(self.system_dict["states"]["resume_train"]):
                if(os.path.isfile(self.system_dict["model_dir_relative"] + 'resume_state-symbol.json')):
                    self.custom_print("    Loading model - {}".format(self.system_dict["model_dir_relative"] + 'resume_state-symbol.json'));
                    self.system_dict["local"]["model"] = load_model(self.system_dict, resume=True);
                else:
                    msg = "Model not found - \"{}\"\n".format(self.system_dict["model_dir_relative"] + 'resume_state-symbol.json');
                    msg += "Training not started. Cannot Run resume Mode";
                    raise ConstraintError(msg);
            else:
                self.custom_print("    Loading pretrained model");
                self.system_dict = setup_model(self.system_dict);
                

            
            self.system_dict = model_to_device(self.system_dict);
            self.custom_print("    Model Loaded on device");
            


            self.system_dict = get_num_layers(self.system_dict);

            if(not self.system_dict["states"]["resume_train"]):
                if(self.system_dict["model"]["type"] == "pretrained"):
                    current_name = "";
                    ip = 0;
                    self.system_dict["local"]["params_to_update"] = [];
                    complete_list = [];
                    for param in self.system_dict["local"]["model"].collect_params().values():
                        if(ip==0):
                            current_name = '_'.join(param.name.split("_")[:-1]);
                            if("running" in current_name):
                                current_name = '_'.join(current_name.split("_")[:-1]);
                            if(param.grad_req == "write"):
                                if(current_name not in complete_list):
                                    complete_list.append(current_name);
                                self.system_dict["local"]["params_to_update"].append(current_name);
                        else:
                            if(current_name != '_'.join(param.name.split("_")[:-1])):
                                current_name = '_'.join(param.name.split("_")[:-1]);
                                if("running" in current_name):
                                    current_name = '_'.join(current_name.split("_")[:-1]);
                                if(param.grad_req == "write"):
                                    if(current_name not in complete_list):
                                        complete_list.append(current_name);
                                    self.system_dict["local"]["params_to_update"].append(current_name);
                        ip += 1;
                    self.system_dict["model"]["params"]["num_params_to_update"] = len(complete_list);
                    self.system_dict["model"]["status"] = True;

                else:
                    current_name = "";
                    ip = 0;
                    self.system_dict["local"]["params_to_update"] = [];
                    complete_list = [];
                    for param in self.system_dict["local"]["model"].collect_params().values():
                        if(ip==0):
                            current_name = param.name.split("_")[0];
                            if("running" in current_name):
                                current_name = '_'.join(current_name.split("_")[:-1]);
                            if(param.grad_req == "write"):
                                if(current_name not in complete_list):
                                    complete_list.append(current_name);
                                self.system_dict["local"]["params_to_update"].append(current_name);
                        else:
                            if(current_name != param.name.split("_")[0]):
                                current_name = param.name.split("_")[0];
                                if("running" in current_name):
                                    current_name = '_'.join(current_name.split("_")[:-1]);
                                if(param.grad_req == "write"):
                                    if(current_name not in complete_list):
                                        complete_list.append(current_name);
                                    self.system_dict["local"]["params_to_update"].append(current_name);
                        ip += 1;
                    self.system_dict["model"]["params"]["num_params_to_update"] = len(complete_list);
                    self.system_dict["model"]["status"] = True;

            if(self.system_dict["model"]["type"] == "custom"):
                self.custom_print("        Model name:                           {}".format("Custom Model"));
            else:
                self.custom_print("        Model name:                           {}".format(self.system_dict["model"]["params"]["model_name"]));
            self.custom_print("        Num of potentially trainable layers:  {}".format(self.system_dict["model"]["params"]["num_layers"]));
            self.custom_print("        Num of actual trainable layers:       {}".format(self.system_dict["model"]["params"]["num_params_to_update"]));
            self.custom_print("");
            
    ###############################################################################################################################################
