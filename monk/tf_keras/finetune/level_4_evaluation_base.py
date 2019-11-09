from tf_keras.finetune.imports import *
from system.imports import *

from tf_keras.finetune.level_3_training_base import finetune_training


class finetune_evaluation(finetune_training):
    @accepts("self", verbose=int, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose);


    ###############################################################################################################################################
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def set_evaluation_final(self):
        self.custom_print("Testing");
        self.system_dict["testing"]["status"] = False;


        if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
            verbose=1;
        else:
            verbose=0;  

        running_corrects = 0
        class_dict = {};
        class_names = list(self.system_dict["dataset"]["params"]["classes"].keys());

        for i in range(len(class_names)):
            class_dict[class_names[i]] = {};
            class_dict[class_names[i]]["num_images"] = 0;
            class_dict[class_names[i]]["num_correct"] = 0;


        step_size_test=self.system_dict["local"]["data_loaders"]["test"].n//self.system_dict["local"]["data_loaders"]["test"].batch_size


        output = self.system_dict["local"]["model"].predict_generator(generator=self.system_dict["local"]["data_loaders"]["test"], 
                                steps=step_size_test, 
                                callbacks=None, 
                                max_queue_size=10, 
                                workers=psutil.cpu_count(), 
                                use_multiprocessing=False, 
                                verbose=verbose);


        i = 0;
        labels = self.system_dict["local"]["data_loaders"]["test"].labels;
        for i in range(len(labels)):
            gt = class_names[labels[i]];
            l = class_names[np.argmax(output[i])];
            class_dict[gt]["num_images"] += 1;

            if(l==gt):
                running_corrects += 1;
                class_dict[gt]["num_correct"] += 1;


        accuracy = running_corrects / len(self.system_dict["local"]["data_loaders"]['test'].labels);


        self.custom_print("");
        self.custom_print("    Result");
        self.custom_print("        class based accuracies");
        for i in range(len(class_names)):
            self.custom_print("            {}. {} - {} %".format(i, class_names[i], 
                class_dict[class_names[i]]["num_correct"]/class_dict[class_names[i]]["num_images"]*100));
            class_dict[class_names[i]]["accuracy(%)"] = class_dict[class_names[i]]["num_correct"]/class_dict[class_names[i]]["num_images"]*100;
        self.custom_print("        total images:            {}".format(len(self.system_dict["local"]["data_loaders"]["test"])));
        self.custom_print("        num correct predictions: {}".format(running_corrects));
        self.custom_print("        Average accuracy (%):    {}".format(accuracy*100));

        self.system_dict["testing"]["num_images"] = len(self.system_dict["local"]["data_loaders"]["test"]);
        self.system_dict["testing"]["num_correct_predictions"] = running_corrects;
        self.system_dict["testing"]["percentage_accuracy"] = accuracy*100;
        self.system_dict["testing"]["class_accuracy"] = class_dict
        self.system_dict["testing"]["status"] = True;
        self.custom_print("");
        
        return accuracy*100, class_dict;

    ###############################################################################################################################################





    ###############################################################################################################################################
    @accepts("self", img_name=[str, bool], img_dir=[str, bool], return_raw=bool, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def set_prediction_final(self, img_name=False, img_dir=False, return_raw=False):
        self.custom_print("Prediction");

        if(not self.system_dict["dataset"]["params"]["input_size"]):
            msg = "Input Size not set for experiment.\n";
            msg += "Tip: Use update_input_size";
            raise ConstraintError(msg);

        self.system_dict = set_transform_test(self.system_dict);




        if(not self.system_dict["dataset"]["params"]["classes"]):
            msg = "Class information unavailabe.\n";
            msg += "Labels returned - Indexes instead of classes";
            ConstraintWarning(msg);


        if(img_name):
            self.custom_print("    Image name:         {}".format(img_name));

            label, score, raw_output = process_single(img_name, return_raw, self.system_dict);

            self.custom_print("    Predicted class:      {}".format(label));
            self.custom_print("    Predicted score:      {}".format(score));
            tmp = {};
            tmp["img_name"] = img_name;
            tmp["predicted_class"] = label;
            tmp["score"] = score;
            if(return_raw):
                tmp["raw"] = raw_output;
            self.custom_print("");                
            return tmp;

        if(img_dir):
            output = [];
            self.custom_print("    Dir path:           {}".format(img_dir));
            img_list = os.listdir(img_dir);
            self.custom_print("    Total Images:       {}".format(len(img_list)));
            self.custom_print("Processing Images");
            if(self.system_dict["verbose"]):
                pbar = tqdm(total=len(img_list));


            for i in range(len(img_list)):
                if(self.system_dict["verbose"]):
                    pbar.update();

                img_name = img_dir + "/" + img_list[i];
                
                label, score, raw_output = process_single(img_name, return_raw, self.system_dict);

                tmp = {};
                tmp["img_name"] = img_list[i];
                tmp["predicted_class"] = label;
                tmp["score"] = score;
                if(return_raw):
                    tmp["raw"] = raw_output;
                output.append(tmp);
            self.custom_print("");

            return output
    ###############################################################################################################################################