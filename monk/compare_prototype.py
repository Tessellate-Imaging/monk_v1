from system.imports import *

from system.base_class import system
from system.common import read_json
from system.graphs.line import training_accuracy_curve
from system.graphs.line import validation_accuracy_curve
from system.graphs.line import training_loss_curve
from system.graphs.line import validation_loss_curve
from system.graphs.bar import training_time_plot
from system.graphs.bar import max_accuracy_plot
from system.graphs.bar import max_gpu_usage_plot




class compare(system):
    @accepts("self", verbose=[int, bool, str], post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose)
    

    @error_checks(None, ["name", ["A-Z", "a-z", "0-9", "-", "_"]], post_trace=True)
    @accepts("self", str, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Comparison(self, comparison_name):
        self.set_system_comparison(comparison_name); 
        self.custom_print("Comparison: - {}".format(comparison_name));


    @accepts("self", str, str, post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Add_Experiment(self, project_name, experiment_name):
        json_file = self.system_dict["master_systems_dir_relative"] + project_name + "/" + experiment_name + "/experiment_state.json";
        if(not os.path.isfile(json_file)):
            msg = "Project - {}, Experiment - {} does not exist".format(project_name, experiment_name)
            raise ConstraintError(msg)
        self.system_dict["local"]["experiments_list"].append(json_file);
        self.system_dict["local"]["project_experiment_list"].append(project_name + ":" + experiment_name);
        self.custom_print("Project - {}, Experiment - {} added".format(project_name, experiment_name));


    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=True, trace_rv=True)
    def Generate_Statistics(self):
        self.custom_print("Generating statistics...");
        data = [];
        for i in range(len(self.system_dict["local"]["experiments_list"])):
            fname = self.system_dict["local"]["experiments_list"][i];
            system_dict = read_json(fname);
            data.append(system_dict);


        training_accuracy_curve(data, self.system_dict);
        validation_accuracy_curve(data, self.system_dict);
        training_loss_curve(data, self.system_dict);
        validation_loss_curve(data, self.system_dict);        
        
        training_time_plot(data, self.system_dict);
        max_accuracy_plot(data, self.system_dict);
        max_gpu_usage_plot(data, self.system_dict);
        
        
        # table
        table = [];

        #headers
        headers = ["project", "experiment", "base_model", "origin", "best val acc", "test acc", "num test images", \
        "epochs", "base lr", "optimizer", "lr scheduler", "loss func", "All layers trained", "gpu used", "max gpu usage", "training time", "train dataset type", \
        "num train images", "num val images", "shuffled dataset", "train transforms", "val transforms", "test transforms"] 


        for i in range(len(data)):
            tmp = [];
            tmp.append(str(data[i]["project_name"]));
            tmp.append(str(data[i]["experiment_name"]));
            model_name = data[i]["model"]["params"]["model_name"];
            if("/" in model_name):
                model_name = model_name.split("/")[-1];
            tmp.append(str(model_name));
            tmp.append(str(data[i]["origin"]));
            if(data[i]["training"]["status"]):
                tmp.append(str(data[i]["training"]["outputs"]["best_val_acc"]));
            else:
                tmp.append("NA");
            if(data[i]["testing"]["status"]):
                tmp.append(str(data[i]["testing"]["percentage_accuracy"]));
                tmp.append(str(data[i]["testing"]["num_images"]));
            else:
                tmp.append("NA");
                tmp.append("NA");
            tmp.append(str(data[i]["hyper-parameters"]["num_epochs"]));
            tmp.append(str(data[i]["hyper-parameters"]["optimizer"]["params"]["lr"]));
            tmp.append(str(data[i]["hyper-parameters"]["optimizer"]["name"]));
            tmp.append(str(data[i]["hyper-parameters"]["learning_rate_scheduler"]["name"]));
            tmp.append(str(data[i]["hyper-parameters"]["loss"]["name"]));
            tmp.append(str(data[i]["model"]["params"]["freeze_base_network"]));
            tmp.append(str(data[i]["model"]["params"]["use_gpu"]));
            if(data[i]["training"]["status"]):
                tmp.append(str(data[i]["training"]["outputs"]["training_time"]));
                tmp.append(str(data[i]["training"]["outputs"]["max_gpu_usage"]));
            else:
                tmp.append("NA");
                tmp.append("NA");
            tmp.append(str(data[i]["dataset"]["dataset_type"]));
            tmp.append(str(data[i]["dataset"]["params"]["num_train_images"]));
            tmp.append(str(data[i]["dataset"]["params"]["num_val_images"]));
            tmp.append(str(data[i]["dataset"]["params"]["train_shuffle"]));
            tmp.append(str(data[i]["dataset"]["transforms"]["train"]));
            tmp.append(str(data[i]["dataset"]["transforms"]["val"]));
            tmp.append(str(data[i]["dataset"]["transforms"]["test"]));
            table.append(tmp);

        my_df = pd.DataFrame(table);
        fname =  self.system_dict["master_comparison_dir_relative"] + "comparison.csv";
        my_df.to_csv(fname, index=False, header=headers);

        self.custom_print("Generated");
        self.custom_print("");