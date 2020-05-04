from gluon.finetune.imports import *
from system.imports import *
from gluon.finetune.level_2_model_base import finetune_model



class finetune_training(finetune_model):
    '''
    Base class for training and associated functions

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
    def get_training_estimate(self):
        '''
        Get estimated time for training a single epoch based on all set parameters

        Args:
            None

        Returns:
            float: Total time per epoch in seconds
        '''
        total_time_per_epoch = 0;

        self.system_dict = load_scheduler(self.system_dict);
        self.system_dict = load_optimizer(self.system_dict);
        self.system_dict = load_loss(self.system_dict);


        num_iterations_train = len(self.system_dict["local"]["data_loaders"]["train"])//10;
        num_iterations_val = len(self.system_dict["local"]["data_loaders"]["val"])//10;


        since = time.time();
        train_loss = 0;
        for i, batch in enumerate(self.system_dict["local"]["data_loaders"]["train"]):
            data = mx.gluon.utils.split_and_load(batch[0], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
            label = mx.gluon.utils.split_and_load(batch[1], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
            with ag.record():
                outputs = [self.system_dict["local"]["model"](X) for X in data]
                loss = [self.system_dict["local"]["criterion"](yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            
            if(i==num_iterations_train):
                break;

        for i, batch in enumerate(self.system_dict["local"]["data_loaders"]["val"]):
            data = mx.gluon.utils.split_and_load(batch[0], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
            label = mx.gluon.utils.split_and_load(batch[1], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
            
            with ag.record():
                outputs = [self.system_dict["local"]["model"](X) for X in data]
                loss = [self.system_dict["local"]["criterion"](yhat, y) for yhat, y in zip(outputs, label)]
            
            
            if(i==num_iterations_val):
                break;

        total_time_per_epoch = (time.time() - since)*10;
        
        return total_time_per_epoch;
    ###############################################################################################################################################





    ###############################################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_training_evaluation(self):
        '''
        Base function for running validation while training

        Args:
            None

        Returns:
            dict: Validation metrics
            float: Test Loss 
        '''
        num_batch = len(self.system_dict["local"]["data_loaders"]["val"]);

        if(self.system_dict["dataset"]["label_type"] == "single"):
            metric = mx.metric.Accuracy();
        else:
            metric = mx.metric.CustomMetric(feval=self.custom_metric)
            
        if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
            pbar = tqdm(total=num_batch);
        
        test_loss = 0;
        
        for i, batch in enumerate(self.system_dict["local"]["data_loaders"]["val"]):
            if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                pbar.update();

            data = mx.gluon.utils.split_and_load(batch[0], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
            label = mx.gluon.utils.split_and_load(batch[1], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
            
            with ag.record():
                outputs = [self.system_dict["local"]["model"](X) for X in data]
                loss = [self.system_dict["local"]["criterion"](yhat, y) for yhat, y in zip(outputs, label)]
            test_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            metric.update(label, outputs)
        return metric.get(), test_loss;
    ###############################################################################################################################################




    ###############################################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_training_final(self):
        '''
        Main training function

        Args:
            None

        Returns:
            None
        '''
        if(self.system_dict["states"]["resume_train"]):
            self.custom_print("Training Resume");
            self.system_dict = load_scheduler(self.system_dict);
            self.system_dict = load_optimizer(self.system_dict);
            self.system_dict = load_loss(self.system_dict);

            if(self.system_dict["dataset"]["label_type"] == "single"):
                metric = mx.metric.Accuracy();
            else:
                metric = mx.metric.CustomMetric(feval=self.custom_metric)

            trainer = mx.gluon.Trainer(self.system_dict["local"]["model"].collect_params(), optimizer=self.system_dict["local"]["optimizer"]);

            self.system_dict["training"]["status"] = False;
            since = time.time()
            pid = os.getpid();

            if(self.system_dict["training"]["settings"]["save_training_logs"]):
                val_acc_history = list(np.load(self.system_dict["log_dir"] + "val_acc_history.npy", allow_pickle=True));
                train_acc_history = list(np.load(self.system_dict["log_dir"] + "train_acc_history.npy", allow_pickle=True));
                val_loss_history = list(np.load(self.system_dict["log_dir"] + "val_loss_history.npy", allow_pickle=True));
                train_loss_history = list(np.load(self.system_dict["log_dir"] + "train_loss_history.npy", allow_pickle=True));

            num_batch_train = len(self.system_dict["local"]["data_loaders"]["train"]);
            num_batch_val = len(self.system_dict["local"]["data_loaders"]["val"]);

            best_acc = 0.0;
            best_acc_epoch = 0;
            max_gpu_usage = 0;

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
                train_loss = 0
                metric.reset()
                
                if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                    pbar = tqdm(total=num_batch_train);
                
                for i, batch in enumerate(self.system_dict["local"]["data_loaders"]["train"]):
                    if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                        pbar.update();

                    data = mx.gluon.utils.split_and_load(batch[0], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
                    label = mx.gluon.utils.split_and_load(batch[1], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
    
                    with ag.record():
                        outputs = [self.system_dict["local"]["model"](X) for X in data]
                        loss = [self.system_dict["local"]["criterion"](yhat, y) for yhat, y in zip(outputs, label)]
                    for l in loss:
                        l.backward()

                    trainer.step(self.system_dict["dataset"]["params"]["batch_size"]);
                    train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
                    metric.update(label, outputs)

                _, train_acc = metric.get()
                train_loss /= num_batch_train;

                val_acc, val_loss = self.set_training_evaluation();
                val_acc = val_acc[1];
                val_loss /= num_batch_val;

                

                if(not os.getcwd() == "/kaggle/working"):
                    if(self.system_dict["model"]["params"]["use_gpu"]):
                        GPUs = GPUtil.getGPUs()
                        gpuMemoryUsed = GPUs[0].memoryUsed
                        if(self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] < int(gpuMemoryUsed)):
                            self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] = int(gpuMemoryUsed);
                else:
                    gpuMemoryUsed = 0;
                    self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] = 0;

                if(self.system_dict["training"]["settings"]["save_training_logs"]):
                    val_acc_history.append(val_acc);
                    val_loss_history.append(val_loss);
                    train_acc_history.append(train_acc);
                    train_loss_history.append(train_loss);

                if(val_acc > best_acc):
                    best_acc = val_acc;
                    best_acc_epoch = epoch;
                    if(self.system_dict["training"]["settings"]["save_intermediate_models"]):
                        self.system_dict["local"]["model"].export(self.system_dict["model_dir"] + self.system_dict["training"]["settings"]["intermediate_model_prefix"],
                            epoch=epoch)
                    self.system_dict["local"]["model"].export(self.system_dict["model_dir"] + "best_model", epoch=0);

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

                    create_train_test_plots_accuracy([train_acc_history, val_acc_history], ["Epoch Num", "Accuracy"], self.system_dict["log_dir"], show_img=False, save_img=True);
                    create_train_test_plots_loss([train_loss_history, val_loss_history], ["Epoch Num", "Loss"], self.system_dict["log_dir"], show_img=False, save_img=True);
                
                self.system_dict["local"]["model"].export(self.system_dict["model_dir"] + "resume_state", epoch=0);
                if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                    self.custom_print("");
                    self.custom_print("");

                if(self.system_dict["training"]["settings"]["display_progress"]):
                    curr_lr = trainer.learning_rate
                    self.custom_print("    curr_lr - {}".format(curr_lr));
                    self.custom_print('    [Epoch %d] Train-acc: %.3f, Train-loss: %.3f | Val-acc: %3f, Val-loss: %.3f, | time: %.1f sec' %
                             (epoch+1, train_acc, train_loss, val_acc, val_loss, time.time() - since));
                    self.custom_print("");
                self.system_dict["training"]["outputs"]["epochs_completed"] = epoch+1;
                save(self.system_dict);    



        elif(self.system_dict["states"]["eval_infer"]):
            msg = "Cannot train in testing (eval_infer) mode.\n";
            msg += "Tip - use new_experiment function with a copy_from argument.\n";
            raise ConstraintError(msg);
       

        else:
            self.custom_print("Training Start");
            self.system_dict = load_scheduler(self.system_dict);
            self.system_dict = load_optimizer(self.system_dict);
            self.system_dict = load_loss(self.system_dict);

            if(self.system_dict["dataset"]["label_type"] == "single"):
                metric = mx.metric.Accuracy();
            else:
                metric = mx.metric.CustomMetric(feval=self.custom_metric)
            trainer = mx.gluon.Trainer(self.system_dict["local"]["model"].collect_params(), optimizer=self.system_dict["local"]["optimizer"]);

            self.system_dict["training"]["status"] = False;

            pid = os.getpid()

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

            for epoch in range(self.system_dict["hyper-parameters"]["num_epochs"]):
                if(self.system_dict["training"]["settings"]["display_progress"]):
                    self.custom_print('    Epoch {}/{}'.format(epoch+1, self.system_dict["hyper-parameters"]["num_epochs"]))
                    self.custom_print('    ' + '-' * 10)

                since = time.time();
                train_loss = 0
                metric.reset()
                
                if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                    pbar = tqdm(total=num_batch_train);
                
                for i, batch in enumerate(self.system_dict["local"]["data_loaders"]["train"]):
                    if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                        pbar.update();

                    data = mx.gluon.utils.split_and_load(batch[0], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
                    label = mx.gluon.utils.split_and_load(batch[1], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
    
                    with ag.record():
                        outputs = [self.system_dict["local"]["model"](X) for X in data]
                        loss = [self.system_dict["local"]["criterion"](yhat, y) for yhat, y in zip(outputs, label)]
                    for l in loss:
                        l.backward()

                    trainer.step(self.system_dict["dataset"]["params"]["batch_size"]);
                    train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
                    metric.update(label, outputs)

                _, train_acc = metric.get()
                train_loss /= num_batch_train;

                val_acc, val_loss = self.set_training_evaluation();
                val_acc = val_acc[1];
                val_loss /= num_batch_val;

                

                if(not os.getcwd() == "/kaggle/working"):
                    if(self.system_dict["model"]["params"]["use_gpu"]):
                        GPUs = GPUtil.getGPUs()
                        gpuMemoryUsed = GPUs[0].memoryUsed
                        if(self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] < int(gpuMemoryUsed)):
                            self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] = int(gpuMemoryUsed);
                else:
                    gpuMemoryUsed = 0;
                    self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] = 0;

                if(self.system_dict["training"]["settings"]["save_training_logs"]):
                    val_acc_history.append(val_acc);
                    val_loss_history.append(val_loss);
                    train_acc_history.append(train_acc);
                    train_loss_history.append(train_loss);

                if(self.system_dict["training"]["settings"]["save_intermediate_models"]):
                    self.system_dict["local"]["model"].export(self.system_dict["model_dir"] + self.system_dict["training"]["settings"]["intermediate_model_prefix"],
                            epoch=epoch)


                if(val_acc > best_acc):
                    best_acc = val_acc;
                    best_acc_epoch = epoch;
                    self.system_dict["local"]["model"].export(self.system_dict["model_dir"] + "best_model", epoch=0);
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

                    create_train_test_plots_accuracy([train_acc_history, val_acc_history], ["Epoch Num", "Accuracy"], self.system_dict["log_dir"], show_img=False, save_img=True);
                    create_train_test_plots_loss([train_loss_history, val_loss_history], ["Epoch Num", "Loss"], self.system_dict["log_dir"], show_img=False, save_img=True);
                
                self.system_dict["local"]["model"].export(self.system_dict["model_dir"] + "resume_state", epoch=0);
                if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                    self.custom_print("");
                    self.custom_print("");

                if(self.system_dict["training"]["settings"]["display_progress"]):
                    curr_lr = trainer.learning_rate
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

            self.system_dict["local"]["model"].export(self.system_dict["model_dir"] + "final", epoch=0);

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






    ###############################################################################################################################################
    def custom_metric(self, labels, raw_scores):
        num_correct = 0;
        total_labels = 0;

        list_classes = [];

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                score = logistic.cdf(raw_scores[i][j])
                pred = False
                if(score > 0.5):
                    pred = True
                else:
                    pred = False
                if(pred and labels[i][j]):
                    num_correct += 1;

                if(labels[i][j]):
                    total_labels += 1;

        return num_correct/total_labels;

    ###############################################################################################################################################


