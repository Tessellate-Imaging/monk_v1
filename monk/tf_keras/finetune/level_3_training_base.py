from tf_keras.finetune.imports import *
from system.imports import *
from tf_keras.finetune.level_2_model_base import finetune_model



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


        self.system_dict = load_scheduler(self.system_dict);
        self.system_dict = load_optimizer(self.system_dict);
        self.system_dict = load_loss(self.system_dict);

        self.system_dict["local"]["model"].compile(optimizer=self.system_dict["local"]["optimizer"], 
                loss=self.system_dict["local"]["criterion"], metrics=['accuracy']);

        time_callback = TimeHistory();
        initial_epoch = 0;

        step_size_estimate = self.system_dict["local"]["data_loaders"]["estimate"].n//self.system_dict["local"]["data_loaders"]["estimate"].batch_size;

        self.system_dict["local"]["model"].fit_generator(generator=self.system_dict["local"]["data_loaders"]["estimate"],
                           steps_per_epoch=step_size_estimate,
                           epochs=1,
                           callbacks=[time_callback],
                           workers=psutil.cpu_count(),
                           initial_epoch = initial_epoch,
                           verbose=0)

        time_taken = time_callback.times[0];
        num_images = len(self.system_dict["local"]["data_loaders"]["estimate"].labels);
        time_taken_per_image = time_taken/num_images;

        total_time_per_epoch = time_taken_per_image*(self.system_dict["dataset"]["params"]["num_train_images"] + 
            self.system_dict["dataset"]["params"]["num_val_images"]);

        return total_time_per_epoch;

    ###############################################################################################################################################
        


    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def set_training_final(self):
        if(self.system_dict["states"]["resume_train"]):
            self.custom_print("Training Resume");
            total_time_per_epoch = 0;

            self.system_dict = load_scheduler(self.system_dict);
            self.system_dict = load_optimizer(self.system_dict);
            self.system_dict = load_loss(self.system_dict);

            self.system_dict["training"]["status"] = False;

            pid = os.getpid();


            log_dir = self.system_dict["log_dir_relative"];
            model_dir = self.system_dict["model_dir_relative"];
            intermediate_model_prefix = self.system_dict["training"]["settings"]["intermediate_model_prefix"];
            save_intermediate_models = self.system_dict["training"]["settings"]["save_intermediate_models"];
            display_progress_realtime = self.system_dict["training"]["settings"]["display_progress_realtime"];
            display_progress = self.system_dict["training"]["settings"]["display_progress"];
            num_epochs = self.system_dict["hyper-parameters"]["num_epochs"];

            f = open(self.system_dict["log_dir"] + "/model_history_log.csv", 'r');
            lines = f.readlines();
            f.close();
            epochs_completed = len(lines)-1;

            if(self.system_dict["training"]["settings"]["save_training_logs"]):
                history_df = pd.read_csv(self.system_dict["log_dir"] + "/model_history_log.csv");
                val_acc_history = history_df['val_acc'].tolist();
                train_acc_history = history_df['acc'].tolist();
                val_loss_history = history_df['val_loss'].tolist();
                train_loss_history = history_df['loss'].tolist();

            f = open(self.system_dict["log_dir"] + "/times.txt", 'r');
            lines = f.readlines();
            times_history = [];
            for i in range(len(lines)):
                times_history.append(float(lines[i][:len(lines[i])-1]));


            csv_logger = krc.CSVLogger(log_dir + "model_history_log.csv", append=False);

            if(not self.system_dict["verbose"]):
                verbose=0;
            elif(display_progress_realtime):
                verbose=1;
            elif(display_progress):
                verbose=2;
            else:
                verbose=0;
            

            ckpt_all = krc.ModelCheckpoint(model_dir + intermediate_model_prefix + '{epoch:02d}.h5', monitor='val_loss', verbose=verbose, 
                save_best_only=False, save_weights_only=False, mode='auto', period=1);

            ckpt_best = krc.ModelCheckpoint(model_dir + 'best_model.h5', monitor='val_loss', verbose=verbose, 
                save_best_only=True, save_weights_only=False, mode='auto', period=1);

            resume = krc.ModelCheckpoint(model_dir + 'resume_state.h5', monitor='val_loss', verbose=verbose, 
                save_best_only=False, save_weights_only=False, mode='auto', period=1);
            
            time_callback = TimeHistory(log_dir)
            memory_callback = MemoryHistory();


            callbacks = [krc.History(), memory_callback, time_callback, csv_logger, resume, ckpt_best];

            if(save_intermediate_models):
                callbacks.append(ckpt_all);

            if(self.system_dict["local"]["learning_rate_scheduler"]):
                if(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] == "reduceonplateaulr"):
                    callbacks.append(self.system_dict["local"]["learning_rate_scheduler"]);
                else:
                    callbacks.append(krc.LearningRateScheduler(self.system_dict["local"]["learning_rate_scheduler"]));


            step_size_train = self.system_dict["local"]["data_loaders"]["train"].n//self.system_dict["local"]["data_loaders"]["train"].batch_size;
            step_size_val = self.system_dict["local"]["data_loaders"]["val"].n//self.system_dict["local"]["data_loaders"]["val"].batch_size;


            initial_epoch = epochs_completed;

            


            self.system_dict["local"]["model"].compile(optimizer=self.system_dict["local"]["optimizer"], 
                loss=self.system_dict["local"]["criterion"], metrics=['accuracy']);


            history = self.system_dict["local"]["model"].fit_generator(generator=self.system_dict["local"]["data_loaders"]["train"],
                               steps_per_epoch=step_size_train,
                               epochs=num_epochs,
                               callbacks=callbacks,
                               validation_data=self.system_dict["local"]["data_loaders"]["val"],
                               validation_steps=step_size_val,
                               initial_epoch = initial_epoch,
                               verbose=verbose);

            time_elapsed_since = 0;
            times_history += time_callback.times
            for i in range(len(times_history)):
                time_elapsed_since += times_history[i];


            self.system_dict["training"]["outputs"]["training_time"] = "{:.0f}m {:.0f}s".format(time_elapsed_since // 60, time_elapsed_since % 60);

            if(keras.__version__.split(".")[1] == "3"):
                val_acc_history = history.history['val_accuracy'];
                val_loss_history = history.history['val_loss'];
                train_acc_history = history.history['accuracy'];
                train_loss_history = history.history['loss'];
            else:
                val_acc_history = history.history['val_acc'];
                val_loss_history = history.history['val_loss'];
                train_acc_history = history.history['acc'];
                train_loss_history = history.history['loss'];


            self.system_dict["training"]["outputs"]["best_val_acc"] = max(val_acc_history);
            self.system_dict["training"]["outputs"]["best_val_acc_epoch_num"] = val_acc_history.index(self.system_dict["training"]["outputs"]["best_val_acc"]);
            self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] = memory_callback.max_gpu_usage;


            if(self.system_dict["training"]["settings"]["save_training_logs"]):
                np.save(self.system_dict["log_dir"] + "val_acc_history.npy", np.array(val_acc_history));
                np.save(self.system_dict["log_dir"] + "val_loss_history.npy", np.array(val_loss_history));
                np.save(self.system_dict["log_dir"] + "train_acc_history.npy", np.array(train_acc_history));
                np.save(self.system_dict["log_dir"] + "train_loss_history.npy", np.array(train_loss_history));


            if(self.system_dict["training"]["settings"]["display_progress"]):
                self.custom_print('    Training completed in: {:.0f}m {:.0f}s'.format(time_elapsed_since // 60, time_elapsed_since % 60))
                self.custom_print('    Best val Acc:          {:4f}'.format(self.system_dict["training"]["outputs"]["best_val_acc"]))
                self.custom_print("");



        elif(self.system_dict["states"]["eval_infer"]):
            msg = "Cannot train in testing (eval_infer) mode.\n";
            msg += "Tip - use new_experiment function with a copy_from argument.\n";
            raise ConstraintError(msg);



        else:
            self.custom_print("Training Start");

            self.system_dict = load_scheduler(self.system_dict);
            self.system_dict = load_optimizer(self.system_dict);
            self.system_dict = load_loss(self.system_dict);

            self.system_dict["training"]["status"] = False;

            pid = os.getpid();


            log_dir = self.system_dict["log_dir_relative"];
            model_dir = self.system_dict["model_dir_relative"];
            intermediate_model_prefix = self.system_dict["training"]["settings"]["intermediate_model_prefix"];
            save_intermediate_models = self.system_dict["training"]["settings"]["save_intermediate_models"];
            display_progress_realtime = self.system_dict["training"]["settings"]["display_progress_realtime"];
            display_progress = self.system_dict["training"]["settings"]["display_progress"];
            num_epochs = self.system_dict["hyper-parameters"]["num_epochs"];


            if(not self.system_dict["verbose"]):
                verbose=0;
            elif(display_progress_realtime):
                verbose=1;
            elif(display_progress):
                verbose=2;
            else:
                verbose=0;


            csv_logger = krc.CSVLogger(log_dir + "model_history_log.csv", append=False);
            ckpt_all = krc.ModelCheckpoint(model_dir + intermediate_model_prefix + '{epoch:02d}.h5', monitor='val_loss', verbose=verbose, 
                save_best_only=False, save_weights_only=False, mode='auto', period=1);

            ckpt_best = krc.ModelCheckpoint(model_dir + 'best_model.h5', monitor='val_loss', verbose=verbose, 
                save_best_only=True, save_weights_only=False, mode='auto', period=1);

            resume = krc.ModelCheckpoint(model_dir + 'resume_state.h5', monitor='val_loss', verbose=verbose, 
                save_best_only=False, save_weights_only=False, mode='auto', period=1);
            
            time_callback = TimeHistory(log_dir)
            memory_callback = MemoryHistory();


            callbacks = [krc.History(), memory_callback, time_callback, csv_logger, resume, ckpt_best];

            if(save_intermediate_models):
                callbacks.append(ckpt_all);

            if(self.system_dict["local"]["learning_rate_scheduler"]):
                if(self.system_dict["hyper-parameters"]["learning_rate_scheduler"]["name"] == "reduceonplateaulr"):
                    callbacks.append(self.system_dict["local"]["learning_rate_scheduler"]);
                else:
                    callbacks.append(krc.LearningRateScheduler(self.system_dict["local"]["learning_rate_scheduler"]));


            step_size_train = self.system_dict["local"]["data_loaders"]["train"].n//self.system_dict["local"]["data_loaders"]["train"].batch_size;
            step_size_val = self.system_dict["local"]["data_loaders"]["val"].n//self.system_dict["local"]["data_loaders"]["val"].batch_size;


            initial_epoch = 0;

            if(not self.system_dict["verbose"]):
                verbose=0;
            elif(display_progress_realtime):
                verbose=1;
            elif(display_progress):
                verbose=2;
            else:
                verbose=0;


            self.system_dict["local"]["model"].compile(optimizer=self.system_dict["local"]["optimizer"], 
                loss=self.system_dict["local"]["criterion"], metrics=['accuracy']);


            history = self.system_dict["local"]["model"].fit_generator(generator=self.system_dict["local"]["data_loaders"]["train"],
                               steps_per_epoch=step_size_train,
                               epochs=num_epochs,
                               callbacks=callbacks,
                               validation_data=self.system_dict["local"]["data_loaders"]["val"],
                               validation_steps=step_size_val,
                               initial_epoch = initial_epoch,
                               verbose=verbose);



            time_elapsed_since = 0;
            for i in range(len(time_callback.times)):
                time_elapsed_since += time_callback.times[i];


            self.system_dict["training"]["outputs"]["training_time"] = "{:.0f}m {:.0f}s".format(time_elapsed_since // 60, time_elapsed_since % 60);

            if(keras.__version__.split(".")[1] == "3"):
                val_acc_history = history.history['val_accuracy'];
                val_loss_history = history.history['val_loss'];
                train_acc_history = history.history['accuracy'];
                train_loss_history = history.history['loss'];
            else:
                val_acc_history = history.history['val_acc'];
                val_loss_history = history.history['val_loss'];
                train_acc_history = history.history['acc'];
                train_loss_history = history.history['loss'];

            self.system_dict["training"]["outputs"]["best_val_acc"] = max(val_acc_history);
            self.system_dict["training"]["outputs"]["best_val_acc_epoch_num"] = val_acc_history.index(self.system_dict["training"]["outputs"]["best_val_acc"]);
            self.system_dict["training"]["outputs"]["max_gpu_memory_usage"] = memory_callback.max_gpu_usage;


            if(self.system_dict["training"]["settings"]["save_training_logs"]):
                np.save(self.system_dict["log_dir"] + "val_acc_history.npy", np.array(val_acc_history));
                np.save(self.system_dict["log_dir"] + "val_loss_history.npy", np.array(val_loss_history));
                np.save(self.system_dict["log_dir"] + "train_acc_history.npy", np.array(train_acc_history));
                np.save(self.system_dict["log_dir"] + "train_loss_history.npy", np.array(train_loss_history));


            if(self.system_dict["training"]["settings"]["display_progress"]):
                self.custom_print('    Training completed in: {:.0f}m {:.0f}s'.format(time_elapsed_since // 60, time_elapsed_since % 60))
                self.custom_print('    Best val Acc:          {:4f}'.format(self.system_dict["training"]["outputs"]["best_val_acc"]))
                self.custom_print("");

            

        self.system_dict["local"]["model"].save(self.system_dict["model_dir_relative"] + "final.h5");
        self.system_dict["training"]["status"] = True;
        save(self.system_dict);



        if(not self.system_dict["states"]["eval_infer"]):
            self.custom_print("Training End");
            self.custom_print("");
            self.system_dict["training"]["outputs"]["best_val_acc"] = "{:4f}".format(self.system_dict["training"]["outputs"]["best_val_acc"]);
            self.system_dict["training"]["outputs"]["best_val_acc_epoch_num"] = self.system_dict["training"]["outputs"]["best_val_acc_epoch_num"];
            self.system_dict["training"]["outputs"]["training_time"] = "{:.0f}m {:.0f}s".format(time_elapsed_since // 60, time_elapsed_since % 60);
            self.system_dict["training"]["outputs"]["max_gpu_usage"] = str(self.system_dict["training"]["outputs"]["max_gpu_memory_usage"]) + " Mb";


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
