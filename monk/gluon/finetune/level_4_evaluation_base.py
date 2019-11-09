from gluon.finetune.imports import *
from system.imports import *

from gluon.finetune.level_3_training_base import finetune_training


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
            pbar=tqdm(total=len(self.system_dict["local"]["data_loaders"]["test"]));

        running_corrects = 0
        class_dict = {};
        for i in range(len(self.system_dict["dataset"]["params"]["classes"])):
            class_dict[self.system_dict["dataset"]["params"]["classes"][i]] = {};
            class_dict[self.system_dict["dataset"]["params"]["classes"][i]]["num_images"] = 0;
            class_dict[self.system_dict["dataset"]["params"]["classes"][i]]["num_correct"] = 0;

        for i, batch in enumerate(self.system_dict["local"]["data_loaders"]["test"]):
            if(self.system_dict["training"]["settings"]["display_progress_realtime"] and self.system_dict["verbose"]):
                pbar.update();
            data = mx.gluon.utils.split_and_load(batch[0], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
            label = mx.gluon.utils.split_and_load(batch[1], ctx_list=self.system_dict["local"]["ctx"], batch_axis=0, even_split=False)
            outputs = [self.system_dict["local"]["model"](X) for X in data]
            label = self.system_dict["dataset"]["params"]["classes"][int(label[0].asnumpy())];
            pred = self.system_dict["dataset"]["params"]["classes"][np.argmax(outputs[0].asnumpy())];


            #self.custom_print(outputs[0].shape, type(outputs[0]), label, pred);
            class_dict[label]["num_images"] += 1;
            if(label == pred):
                class_dict[label]["num_correct"] += 1;
                running_corrects += 1;


        accuracy = running_corrects / len(self.system_dict["local"]["data_loaders"]["test"]);
        self.custom_print("");
        self.custom_print("    Result");
        self.custom_print("        class based accuracies");
        for i in range(len(self.system_dict["dataset"]["params"]["classes"])):
            self.custom_print("            {}. {} - {} %".format(i, self.system_dict["dataset"]["params"]["classes"][i], 
                class_dict[self.system_dict["dataset"]["params"]["classes"][i]]["num_correct"]/class_dict[self.system_dict["dataset"]["params"]["classes"][i]]["num_images"]*100));
            class_dict[self.system_dict["dataset"]["params"]["classes"][i]]["accuracy(%)"] = class_dict[self.system_dict["dataset"]["params"]["classes"][i]]["num_correct"]/class_dict[self.system_dict["dataset"]["params"]["classes"][i]]["num_images"]*100;
        self.custom_print("        total images:            {}".format(len(self.system_dict["local"]["data_loaders"]["test"])));
        self.custom_print("        num correct predictions: {}".format(int(running_corrects)));
        self.custom_print("        Average accuracy (%):    {}".format(accuracy*100));

        self.system_dict["testing"]["num_images"] = len(self.system_dict["local"]["data_loaders"]["test"]);
        self.system_dict["testing"]["num_correct_predictions"] = int(running_corrects);
        self.system_dict["testing"]["percentage_accuracy"] = accuracy*100
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
