from monk.system.graphs.imports import *
from monk.system.imports import * 


@accepts([list, type(np.array([1, 2]))], [list, type(np.array([1, 2]))], str, show_img=bool, save_img=bool, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=True)
def create_train_test_plots_accuracy(plots, labels, log_dir, show_img=False, save_img=False):
    '''
    Create line graphs for training and testing accuracies

    Args:
        plots (list): plots[0] contains training data
                        plots[1] contains validation data
        labels (list): Coresponding X-Axis labels
        log_dir (str): Path to folder to save plot 
        show_img (bool): If True, plot is displayed
        save_img (bool): If True, plot is saved.
        verbose (str): If True, prints logs for analysis 

    Returns:
        None
    '''
    plt.plot(plots[0], marker='o', label='Training')
    plt.plot(plots[1], marker='x', label='Validation')
    plt.gca().legend(('Training','Validation'))
    plt.xlabel(labels[0]);
    plt.ylabel(labels[1]);
    file_name = log_dir + "train_val_accuracy.png" 
    plt.savefig(file_name)
    plt.clf()


@accepts([list, type(np.array([1, 2]))], [list, type(np.array([1, 2]))], str, show_img=bool, save_img=bool, post_trace=False)
#@TraceFunction(trace_args=True, trace_rv=True)
def create_train_test_plots_loss(plots, labels, log_dir, show_img=False, save_img=False):
    '''
    Create line graphs for training and testing losses

    Args:
        plots (list): plots[0] contains training data
                        plots[1] contains validation data
        labels (list): Coresponding X-Axis labels
        log_dir (str): Path to folder to save plot 
        show_img (bool): If True, plot is displayed
        save_img (bool): If True, plot is saved.
        verbose (str): If True, prints logs for analysis 

    Returns:
        None
    '''
    plt.plot(plots[0], marker='o', label='Training')
    plt.plot(plots[1], marker='x', label='Validation')
    plt.gca().legend(('Training','Validation'))
    plt.xlabel(labels[0]);
    plt.ylabel(labels[1]);
    file_name = log_dir + "train_val_loss.png" 
    plt.savefig(file_name)
    plt.clf()


@accepts([list, type(np.array([1, 2]))], dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=True)
def training_accuracy_curve(data, system_dict):
    '''
    Create training accuracy line graph

    Args:
        data (list): Entire data
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        None
    '''
    plt.figure(figsize=(20,10))
    for i in range(len(data)):
        if(not data[i]["training"]["status"]):
            print("Training for {} - Incomplete".format(self.project_experiment_list[i]));
            value = np.zeros(1);
            xkcd = mcd.XKCD_COLORS["xkcd:" + overlap[i]].upper()
            plt.plot(value, marker='o', label=system_dict["local"]["project_experiment_list"][i], color=xkcd, linewidth=5.0);
        else:
            value = np.load(data[i]["training"]["outputs"]["log_train_acc_history_relative"], allow_pickle=True);
            xkcd = mcd.XKCD_COLORS["xkcd:" + overlap[i]].upper()
            plt.plot(value, marker='o', label=system_dict["local"]["project_experiment_list"][i], color=xkcd, linewidth=5.0);
    plt.gca().legend(tuple(system_dict["local"]["project_experiment_list"]));
    plt.xlabel("Epoch Num");
    plt.ylabel("Accuracy");
    plt.title("Training Accuracy Curve");
    plt.autoscale()
    file_name = system_dict["comparison_dir"] + "train_accuracy.png" 
    plt.savefig(file_name)
    plt.clf()


@accepts([list, type(np.array([1, 2]))], dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=True)
def validation_accuracy_curve(data, system_dict):
    '''
    Create validation accuracy line graph

    Args:
        data (list): Entire data
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        None
    '''
    plt.figure(figsize=(20,10))
    for i in range(len(data)):
        if(not data[i]["training"]["status"]):
            value = np.zeros(1);
            xkcd = mcd.XKCD_COLORS["xkcd:" + overlap[i]].upper()
            plt.plot(value, marker='o', label=system_dict["local"]["project_experiment_list"][i], color=xkcd, linewidth=5.0);
        else:
            value = np.load(data[i]["training"]["outputs"]["log_val_acc_history_relative"], allow_pickle=True);
            xkcd = mcd.XKCD_COLORS["xkcd:" + overlap[i]].upper()
            plt.plot(value, marker='x', label=system_dict["local"]["project_experiment_list"][i], color=xkcd, linewidth=5.0)
    plt.gca().legend(tuple(system_dict["local"]["project_experiment_list"]));
    plt.xlabel("Epoch Num");
    plt.ylabel("Accuracy");
    plt.title("Validation Accuracy Curve");
    plt.autoscale()
    file_name = system_dict["comparison_dir"] + "val_accuracy.png" 
    plt.savefig(file_name)
    plt.clf()


@accepts([list, type(np.array([1, 2]))], dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=True)
def training_loss_curve(data, system_dict):
    '''
    Create training loss line graph

    Args:
        data (list): Entire data
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        None
    '''
    plt.figure(figsize=(20,10))
    for i in range(len(data)):
        if(not data[i]["training"]["status"]):
            value = np.zeros(1);
            xkcd = mcd.XKCD_COLORS["xkcd:" + overlap[i]].upper()
            plt.plot(value, marker='o', label=system_dict["local"]["project_experiment_list"][i], color=xkcd, linewidth=5.0);
        else:
            value = np.load(data[i]["training"]["outputs"]["log_train_loss_history_relative"], allow_pickle=True);
            xkcd = mcd.XKCD_COLORS["xkcd:" + overlap[i]].upper()
            plt.plot(value, marker='o', label=system_dict["local"]["project_experiment_list"][i], color=xkcd, linewidth=5.0)
    plt.gca().legend(tuple(system_dict["local"]["project_experiment_list"]));
    plt.xlabel("Epoch Num");
    plt.ylabel("Loss");
    plt.title("Training Loss Curve");
    plt.autoscale()
    file_name = system_dict["comparison_dir"] + "train_loss.png" 
    plt.savefig(file_name)
    plt.clf()


@accepts([list, type(np.array([1, 2]))], dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=True)
def validation_loss_curve(data, system_dict):
    '''
    Create validation loss line graph

    Args:
        data (list): Entire data
        system_dict (dict): System dictionary storing experiment state and set variables

    Returns:
        None
    '''
    plt.figure(figsize=(20,10))
    for i in range(len(data)):
        if(not data[i]["training"]["status"]):
            value = np.zeros(1);
            xkcd = mcd.XKCD_COLORS["xkcd:" + overlap[i]].upper()
            plt.plot(value, marker='o', label=system_dict["local"]["project_experiment_list"][i], color=xkcd, linewidth=5.0);
        else:
            value = np.load(data[i]["training"]["outputs"]["log_val_loss_history_relative"], allow_pickle=True);
            xkcd = mcd.XKCD_COLORS["xkcd:" + overlap[i]].upper()
            plt.plot(value, marker='o', label=system_dict["local"]["project_experiment_list"][i], color=xkcd, linewidth=5.0)
    plt.gca().legend(tuple(system_dict["local"]["project_experiment_list"]));
    plt.xlabel("Epoch Num");
    plt.ylabel("Loss");
    plt.title("Validation Loss Curve");
    plt.autoscale()
    file_name = system_dict["comparison_dir"] + "val_loss.png" 
    plt.savefig(file_name)
    plt.clf()




