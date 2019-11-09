from pytorch.finetune.imports import *
from system.imports import *
from pytorch.finetune.level_2_model_base import finetune_model



class finetune_training(finetune_model):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);



    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def get_training_estimate(self):
        total_time_per_epoch = 0;

        self.system_dict = load_optimizer(self.system_dict);
        self.system_dict = load_scheduler(self.system_dict);
        self.system_dict = load_loss(self.system_dict);

        since = time.time();

        

        for phase in ['train', 'val']:
            if phase == 'train':
                self.system_dict["local"]["model"].train() 
            else:
                self.system_dict["local"]["model"].eval()  

            running_loss = 0.0
            running_corrects = 0

            required_iters = len(self.system_dict["local"]["data_loaders"][phase])//10;
            current_iter = 0;

            for inputs, labels in self.system_dict["local"]["data_loaders"][phase]:

                inputs = inputs.to(self.system_dict["local"]["device"]);
                labels = labels.to(self.system_dict["local"]["device"]);

                self.system_dict["local"]["optimizer"].zero_grad();


                with torch.set_grad_enabled(phase == 'train'):
                    if "inception" in self.system_dict["model"]["params"]["model_name"] and phase == 'train':
                        outputs, aux_outputs = self.system_dict["local"]["model"](inputs)
                        loss1 = self.system_dict["local"]["criterion"](outputs, labels)
                        loss2 = self.system_dict["local"]["criterion"](aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = self.system_dict["local"]["model"](inputs)
                        loss = self.system_dict["local"]["criterion"](outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        self.system_dict["local"]["optimizer"].step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                current_iter += 1;
                if(current_iter >= required_iters):
                    break;




        total_time_per_epoch = (time.time() - since)*10;
        
        return total_time_per_epoch;

    ###############################################################################################################################################





    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def set_training_final(self):
        if(self.system_dict["states"]["resume_train"]):
            self.custom_print("Training Resume");
            total_time_per_epoch = 0;

            self.system_dict = load_optimizer(self.system_dict);
            self.system_dict = load_scheduler(self.system_dict);
            self.system_dict = load_loss(self.system_dict);
            
            self.system_dict["training"]["status"] = False;

            pid = os.getpid();

            if(self.system_dict["training"]["settings"]["save_training_logs"]):
                val_acc_history = list(np.load(self.system_dict["log_dir"] + "val_acc_history.npy", allow_pickle=True));
                train_acc_history = list(np.load(self.system_dict["log_dir"] + "train_acc_history.npy", allow_pickle=True));
                val_loss_history = list(np.load(self.system_dict["log_dir"] + "val_loss_history.npy", allow_pickle=True));
                train_loss_history = list(np.load(self.system_dict["log_dir"] + "train_loss_history.npy", allow_pickle=True));

            best_acc = 0.0;
            best_acc_epoch = 0;
            max_gpu_usage = 0;
            best_model_wts = copy.deepcopy(self.system_dict["local"]["model"].state_dict());

            for epoch in range(self.system_dict["hyper-parameters"]["num_epochs"]):
                if(self.system_dict["training"]["settings"]["display_progress"]):
                    self.custom_print('    Epoch {}/{}'.format(epoch+1, self.system_dict["hyper-parameters"]["num_epochs"]))
                    self.custom_print('    ' + '-' * 10)


                if(epoch < self.system_dict["training"]["outputs"]["epochs_completed"]):
                    self.custom_print("Skipping Current Epoch");
                    self.custom_print("");
                    self.custom_print("");
                    continue;

                since = time.time();

                for phase in ['train', 'val']:
                    if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                        pbar=tqdm(total=len(self.system_dict["local"]["data_loaders"][phase]));

                    if phase == 'train':
                        self.system_dict["local"]["model"].train() 
                    else:
                        self.system_dict["local"]["model"].eval()  

                    running_loss = 0.0
                    running_corrects = 0


                    for inputs, labels in self.system_dict["local"]["data_loaders"][phase]:
                        if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                            pbar.update();
                        inputs = inputs.to(self.system_dict["local"]["device"]);
                        labels = labels.to(self.system_dict["local"]["device"]);

                        self.system_dict["local"]["optimizer"].zero_grad();


                        with torch.set_grad_enabled(phase == 'train'):
                            if "inception" in self.system_dict["model"]["params"]["model_name"] and phase == 'train':
                                outputs, aux_outputs = self.system_dict["local"]["model"](inputs)
                                loss1 = self.system_dict["local"]["criterion"](outputs, labels)
                                loss2 = self.system_dict["local"]["criterion"](aux_outputs, labels)
                                loss = loss1 + 0.4*loss2
                            else:
                                outputs = self.system_dict["local"]["model"](inputs)
                                loss = self.system_dict["local"]["criterion"](outputs, labels)

                            _, preds = torch.max(outputs, 1)

                            if phase == 'train':
                                loss.backward()
                                self.system_dict["local"]["optimizer"].step()


                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)


                    epoch_loss = running_loss / len(self.system_dict["local"]["data_loaders"][phase].dataset)
                    epoch_acc = running_corrects.double() / len(self.system_dict["local"]["data_loaders"][phase].dataset)


                    if(self.system_dict["model"]["params"]["use_gpu"]):
                        GPUs = GPUtil.getGPUs()
                        gpuMemoryUsed = GPUs[0].memoryUsed
                        if(self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] < int(gpuMemoryUsed)):
                            self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] = int(gpuMemoryUsed);


                    if(self.system_dict["training"]["settings"]["save_training_logs"]):
                        if phase == 'val':
                            val_acc = epoch_acc;
                            val_loss = epoch_loss;
                            val_acc_history.append(epoch_acc);
                            val_loss_history.append(epoch_loss);
                        else:
                            train_acc = epoch_acc;
                            train_loss = epoch_loss;
                            train_acc_history.append(epoch_acc);
                            train_loss_history.append(epoch_loss);

                if(self.system_dict["training"]["settings"]["save_intermediate_models"]):
                    torch.save(self.system_dict["local"]["model"], self.system_dict["model_dir"] + 
                        self.system_dict["training"]["settings"]["intermediate_model_prefix"] + "{}".format(epoch));



                if(val_acc > best_acc):
                    best_acc = val_acc;
                    best_acc_epoch = epoch;
                    best_model_wts = copy.deepcopy(self.system_dict["local"]["model"].state_dict());
                    torch.save(self.system_dict["local"]["model"], self.system_dict["model_dir"] + "best_model");
                    self.system_dict["training"]["outputs"]["best_val_acc"] = "{:4f}".format(best_acc);
                    self.system_dict["training"]["outputs"]["best_val_acc_epoch_num"] = best_acc_epoch;

                time_elapsed_since = time.time() - since;
                if("training_time" in self.system_dict["training"]["outputs"].keys()):
                    minutes, seconds = self.system_dict["training"]["outputs"]["training_time"].split(" ");
                    minutes = int(minutes[:len(minutes)-1]);
                    seconds = int(seconds[:len(seconds)-1]);
                    time_elapsed_since += minutes*60 + seconds;
                self.system_dict["training"]["outputs"]["training_time"] = "{:.0f}m {:.0f}s".format(time_elapsed_since // 60, time_elapsed_since % 60);


                if(self.system_dict["training"]["settings"]["save_training_logs"]):
                    np.save(self.system_dict["log_dir"] + "val_acc_history.npy", np.array(val_acc_history));
                    np.save(self.system_dict["log_dir"] + "val_loss_history.npy", np.array(val_loss_history));
                    np.save(self.system_dict["log_dir"] + "train_acc_history.npy", np.array(train_acc_history));
                    np.save(self.system_dict["log_dir"] + "train_loss_history.npy", np.array(train_loss_history));
                
                torch.save(self.system_dict["local"]["model"], self.system_dict["model_dir"] + "resume_state");

                if(self.system_dict["local"]["learning_rate_scheduler"]):
                    if(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] == "reduceonplateaulr"):
                        self.system_dict["local"]["learning_rate_scheduler"].step(epoch_loss);
                    else:
                        self.system_dict["local"]["learning_rate_scheduler"].step();

                if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                    self.custom_print("");
                    self.custom_print("");

                if(self.system_dict["training"]["settings"]["display_progress"]):
                    for param_group in self.system_dict["local"]["optimizer"].param_groups:
                        curr_lr = param_group['lr'];
                    self.custom_print("    curr_lr - {}".format(curr_lr));
                    self.custom_print('    [Epoch %d] Train-acc: %.3f, Train-loss: %.3f | Val-acc: %3f, Val-loss: %.3f, | time: %.1f sec' %
                             (epoch+1, train_acc, train_loss, val_acc, val_loss, time.time() - since));
                    self.custom_print("");
                self.system_dict["training"]["outputs"]["epochs_completed"] = epoch+1;
                save(self.system_dict);

            if(self.system_dict["training"]["settings"]["display_progress"]):
                self.custom_print('    Training completed in: {:.0f}m {:.0f}s'.format(time_elapsed_since // 60, time_elapsed_since % 60))
                self.custom_print('    Best val Acc:          {:4f}'.format(best_acc))
                self.custom_print("");
            
            






        
        elif(self.system_dict["states"]["eval_infer"]):
            msg = "Cannot train in testing (eval_infer) mode.\n";
            msg += "Tip - use new_experiment function with a copy_from argument.\n";
            raise ConstraintError(msg);



        else:
            self.custom_print("Training Start");
            self.system_dict = load_optimizer(self.system_dict);
            self.system_dict = load_scheduler(self.system_dict);
            self.system_dict = load_loss(self.system_dict);


            self.system_dict["training"]["status"] = False;

            pid = os.getpid();

            if(self.system_dict["training"]["settings"]["save_training_logs"]):
                val_acc_history = [];
                train_acc_history = [];
                val_loss_history = [];
                train_loss_history = [];

            num_batch_train = len(self.system_dict["local"]["data_loaders"]["train"]);
            num_batch_val = len(self.system_dict["local"]["data_loaders"]["val"]);

            best_acc = 0.0;
            best_acc_epoch = 0;
            max_gpu_usage = 0;
            best_model_wts = copy.deepcopy(self.system_dict["local"]["model"].state_dict());

            for epoch in range(self.system_dict["hyper-parameters"]["num_epochs"]):
                if(self.system_dict["training"]["settings"]["display_progress"]):
                    self.custom_print('    Epoch {}/{}'.format(epoch+1, self.system_dict["hyper-parameters"]["num_epochs"]))
                    self.custom_print('    ' + '-' * 10)

                since = time.time();

                for phase in ['train', 'val']:
                    if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                        pbar=tqdm(total=len(self.system_dict["local"]["data_loaders"][phase]));

                    if phase == 'train':
                        self.system_dict["local"]["model"].train() 
                    else:
                        self.system_dict["local"]["model"].eval()  

                    running_loss = 0.0
                    running_corrects = 0


                    for inputs, labels in self.system_dict["local"]["data_loaders"][phase]:
                        if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                            pbar.update();
                        inputs = inputs.to(self.system_dict["local"]["device"]);
                        labels = labels.to(self.system_dict["local"]["device"]);

                        self.system_dict["local"]["optimizer"].zero_grad();


                        with torch.set_grad_enabled(phase == 'train'):
                            if "inception" in self.system_dict["model"]["params"]["model_name"] and phase == 'train':
                                outputs, aux_outputs = self.system_dict["local"]["model"](inputs)
                                loss1 = self.system_dict["local"]["criterion"](outputs, labels)
                                loss2 = self.system_dict["local"]["criterion"](aux_outputs, labels)
                                loss = loss1 + 0.4*loss2
                            else:
                                outputs = self.system_dict["local"]["model"](inputs)
                                loss = self.system_dict["local"]["criterion"](outputs, labels)

                            _, preds = torch.max(outputs, 1)

                            if phase == 'train':
                                loss.backward()
                                self.system_dict["local"]["optimizer"].step()


                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)


                    epoch_loss = running_loss / len(self.system_dict["local"]["data_loaders"][phase].dataset)
                    epoch_acc = running_corrects.double() / len(self.system_dict["local"]["data_loaders"][phase].dataset)


                    if(self.system_dict["model"]["params"]["use_gpu"]):
                        GPUs = GPUtil.getGPUs()
                        gpuMemoryUsed = GPUs[0].memoryUsed
                        if(self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] < int(gpuMemoryUsed)):
                            self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] = int(gpuMemoryUsed);


                    if(self.system_dict["training"]["settings"]["save_training_logs"]):
                        if phase == 'val':
                            val_acc = epoch_acc;
                            val_loss = epoch_loss;
                            val_acc_history.append(epoch_acc);
                            val_loss_history.append(epoch_loss);
                        else:
                            train_acc = epoch_acc;
                            train_loss = epoch_loss;
                            train_acc_history.append(epoch_acc);
                            train_loss_history.append(epoch_loss);


                if(self.system_dict["training"]["settings"]["save_intermediate_models"]):
                    torch.save(self.system_dict["local"]["model"], self.system_dict["model_dir"] + 
                        self.system_dict["training"]["settings"]["intermediate_model_prefix"] + "{}".format(epoch));



                if(val_acc > best_acc):
                    best_acc = val_acc;
                    best_acc_epoch = epoch;
                    best_model_wts = copy.deepcopy(self.system_dict["local"]["model"].state_dict());
                    torch.save(self.system_dict["local"]["model"], self.system_dict["model_dir"] + "best_model");
                    self.system_dict["training"]["outputs"]["best_val_acc"] = "{:4f}".format(best_acc);
                    self.system_dict["training"]["outputs"]["best_val_acc_epoch_num"] = best_acc_epoch;

                time_elapsed_since = time.time() - since;
                if("training_time" in self.system_dict["training"]["outputs"].keys()):
                    minutes, seconds = self.system_dict["training"]["outputs"]["training_time"].split(" ");
                    minutes = int(minutes[:len(minutes)-1]);
                    seconds = int(seconds[:len(seconds)-1]);
                    time_elapsed_since += minutes*60 + seconds;
                self.system_dict["training"]["outputs"]["training_time"] = "{:.0f}m {:.0f}s".format(time_elapsed_since // 60, time_elapsed_since % 60);


                if(self.system_dict["training"]["settings"]["save_training_logs"]):
                    np.save(self.system_dict["log_dir"] + "val_acc_history.npy", np.array(val_acc_history));
                    np.save(self.system_dict["log_dir"] + "val_loss_history.npy", np.array(val_loss_history));
                    np.save(self.system_dict["log_dir"] + "train_acc_history.npy", np.array(train_acc_history));
                    np.save(self.system_dict["log_dir"] + "train_loss_history.npy", np.array(train_loss_history));
                
                torch.save(self.system_dict["local"]["model"], self.system_dict["model_dir"] + "resume_state");

                if(self.system_dict["local"]["learning_rate_scheduler"]):
                    if(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] == "reduceonplateaulr"):
                        self.system_dict["local"]["learning_rate_scheduler"].step(epoch_loss);
                    else:
                        self.system_dict["local"]["learning_rate_scheduler"].step();

                if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                    self.custom_print("");
                    self.custom_print("");

                if(self.system_dict["training"]["settings"]["display_progress"]):
                    for param_group in self.system_dict["local"]["optimizer"].param_groups:
                        curr_lr = param_group['lr'];
                    self.custom_print("    curr_lr - {}".format(curr_lr));
                    self.custom_print('    [Epoch %d] Train-acc: %.3f, Train-loss: %.3f | Val-acc: %3f, Val-loss: %.3f, | time: %.1f sec' %
                             (epoch+1, train_acc, train_loss, val_acc, val_loss, time.time() - since));
                    self.custom_print("");
                self.system_dict["training"]["outputs"]["epochs_completed"] = epoch+1;
                save(self.system_dict);

            if(self.system_dict["training"]["settings"]["display_progress"]):
                self.custom_print('    Training completed in: {:.0f}m {:.0f}s'.format(time_elapsed_since // 60, time_elapsed_since % 60))
                self.custom_print('    Best val Acc:          {:4f}'.format(best_acc))
                self.custom_print("");


        
        if(not self.system_dict["states"]["eval_infer"]):
            self.custom_print("Training End");
            self.custom_print("");
            self.system_dict["training"]["outputs"]["best_val_acc"] = "{:4f}".format(best_acc);
            self.system_dict["training"]["outputs"]["best_val_acc_epoch_num"] = best_acc_epoch;
            self.system_dict["training"]["outputs"]["training_time"] = "{:.0f}m {:.0f}s".format(time_elapsed_since // 60, time_elapsed_since % 60);
            self.system_dict["training"]["outputs"]["max_gpu_usage"] = str(self.system_dict["training"]["outputs"]["max_gpu_memory_usage"]) + " Mb";

            torch.save(self.system_dict["local"]["model"], self.system_dict["model_dir"] + "final");

            if(self.system_dict["training"]["settings"]["save_training_logs"]): 
                self.custom_print("Training Outputs");
                self.custom_print("    Model Dir:   {}".format(self.system_dict["model_dir"]));
                self.custom_print("    Log Dir:     {}".format(self.system_dict["log_dir"]));
                self.custom_print("    Final model: {}".format("final"));
                self.custom_print("    Best model:  {}".format("best_model"));
                self.custom_print("    Log 1 - Validation accuracy history log: {}".format("val_acc_history.npy"));
                self.custom_print("    Log 2 - Validation loss history log:     {}".format("val_loss_history.npy"));
                self.custom_print("    Log 3 - Training accuracy history log:   {}".format("train_acc_history.npy"));
                self.custom_print("    Log 4 - Training loss history log:       {}".format("train_loss_history.npy"));
                self.custom_print("    Log 5 - Training curve:                  {}".format("train_loss_history.npy"));
                self.custom_print("    Log 6 - Validation curve:                {}".format("train_loss_history.npy"));
                self.custom_print("");

                np.save(self.system_dict["log_dir"] + "val_acc_history.npy", np.array(val_acc_history));
                np.save(self.system_dict["log_dir"] + "val_loss_history.npy", np.array(val_loss_history));
                np.save(self.system_dict["log_dir"] + "train_acc_history.npy", np.array(train_acc_history));
                np.save(self.system_dict["log_dir"] + "train_loss_history.npy", np.array(train_loss_history));
                
                self.system_dict["training"]["outputs"]["log_val_acc_history"] = self.system_dict["log_dir"] + "val_acc_history.npy";
                self.system_dict["training"]["outputs"]["log_val_loss_history"] = self.system_dict["log_dir"] + "val_loss_history.npy";
                self.system_dict["training"]["outputs"]["log_train_acc_history"] = self.system_dict["log_dir"] + "train_acc_history.npy";
                self.system_dict["training"]["outputs"]["log_train_loss_history"] = self.system_dict["log_dir"] + "train_loss_history.npy";

                self.system_dict["training"]["outputs"]["log_val_acc_history_relative"] = self.system_dict["log_dir_relative"] + "val_acc_history.npy";
                self.system_dict["training"]["outputs"]["log_val_loss_history_relative"] = self.system_dict["log_dir_relative"] + "val_loss_history.npy";
                self.system_dict["training"]["outputs"]["log_train_acc_history_relative"] = self.system_dict["log_dir_relative"] + "train_acc_history.npy";
                self.system_dict["training"]["outputs"]["log_train_loss_history_relative"] = self.system_dict["log_dir_relative"] + "train_loss_history.npy";


                create_train_test_plots_accuracy([train_acc_history, val_acc_history], ["Epoch Num", "Accuracy"], self.system_dict["log_dir"], show_img=False, save_img=True);
                create_train_test_plots_loss([train_loss_history, val_loss_history], ["Epoch Num", "Loss"], self.system_dict["log_dir"], show_img=False, save_img=True);

            self.system_dict["training"]["status"] = True;
        

    ###############################################################################################################################################