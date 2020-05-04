from system.graphs.imports import *
from system.imports import *



@accepts([list, type(np.array([1, 2]))], [list, type(np.array([1, 2]))], str, show_img=bool, save_img=bool, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=True)
def create_plot(labels, value, log_dir, show_img=False, save_img=False):
    '''
    Create bar plot

    Args:
        labels (list): X - Axis labels
        value (list): Coresponding Y-Axis values
        log_dir (str): Path to folder to save plot 
        show_img (bool): If True, plot is displayed
        save_img (bool): If True, plot is saved.
        verbose (str): If True, prints logs for analysis 

    Returns:
        None
    '''
    objects = tuple(labels)
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, value, align='center', alpha=0.5, width=0.5)
    plt.xticks(y_pos, objects, rotation='vertical', linespacing=0.9, fontsize='xx-small')
    plt.ylabel('Time - Seconds')
    plt.autoscale()
    plt.title("Training time plot");
    plt.tight_layout()

    if(save_img):
        file_name = log_dir + "/classes_per_image.png" 
        plt.savefig(file_name)
    if(show_img):
        plt.show()    
    plt.clf()



@accepts([list, type(np.array([1, 2]))], dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=True)
def training_time_plot(data, system_dict):
    '''
    Create training-time bar plot

    Args:
        data (list): Entire data
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        None
    '''
    labels = [ '\n'.join(wrap(l, 20)) for l in system_dict["local"]["project_experiment_list"] ]
    objects = tuple(labels)
    y_pos = np.arange(len(objects))
    value = []
    for i in range(len(data)):
        if(data[i]["training"]["status"]):
            minutes, seconds = data[i]["training"]["outputs"]["training_time"].split(" ");
            minutes = int(minutes[:len(minutes)-1]);
            seconds = int(seconds[:len(seconds)-1]);
            value.append(minutes*60 + seconds);
        else:
            value.append(0);
    plt.bar(y_pos, value, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('Time - Seconds')
    plt.autoscale()
    plt.title("Training time plot");
    plt.tight_layout()
    file_name = system_dict["comparison_dir"] + "stats_training_time.png" 
    plt.savefig(file_name)
    plt.clf()


@accepts([list, type(np.array([1, 2]))], dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=True)
def max_accuracy_plot(data, system_dict):
    '''
    Create max-accuracy bar plot

    Args:
        data (list): Entire data
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        None
    '''
    labels = [ '\n'.join(wrap(l, 20)) for l in system_dict["local"]["project_experiment_list"] ]
    objects = tuple(labels)
    y_pos = np.arange(len(objects))
    value = []
    for i in range(len(data)):
        if(data[i]["training"]["status"]):
            acc = float(data[i]["training"]["outputs"]["best_val_acc"])*100;
            value.append(acc);
        else:
            print("Testing for {} - Incomplete".format(system_dict["local"]["project_experiment_list"][i]));
            value.append(0);
    plt.bar(y_pos, value, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('Accuracy (%)')
    plt.autoscale()
    plt.title("Best validation accuracy plot");
    plt.tight_layout()
    file_name = system_dict["comparison_dir"] + "stats_best_val_acc.png" 
    plt.savefig(file_name)
    plt.clf()


@accepts([list, type(np.array([1, 2]))], dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=True)
def max_gpu_usage_plot(data, system_dict):
    '''
    Create gpu-usage bar plot

    Args:
        data (list): Entire data
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        None
    '''
    labels = [ '\n'.join(wrap(l, 20)) for l in system_dict["local"]["project_experiment_list"] ]
    objects = tuple(labels)
    y_pos = np.arange(len(objects))
    value = []
    for i in range(len(data)):
        if(data[i]["training"]["status"]):
            acc = int(data[i]["training"]["outputs"]["max_gpu_usage"].split(" ")[0]);
            value.append(acc);
        else:
            value.append(0);
    plt.bar(y_pos, value, align='center', alpha=0.5)
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('Usage (Mb)')
    plt.autoscale()
    plt.title("Max GPU Memory Usage plot");
    plt.tight_layout()
    file_name = system_dict["comparison_dir"] + "stats_max_gpu_usage.png" 
    plt.savefig(file_name)
    plt.clf()