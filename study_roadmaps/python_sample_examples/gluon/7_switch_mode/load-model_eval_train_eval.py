import os
import sys
sys.path.append("../../../monk/");
import psutil

from gluon_prototype import prototype



gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-2");


gtf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet18_v1", freeze_base_network=True, num_epochs=2);




##################################################  Reset Model #######################################################
gtf.reset_model();
#######################################################################################################################



############################################## Load model from external path ##########################################
gtf.update_model_path(["workspace/sample-project-1/sample-experiment-1/output/models/resume_state-symbol.json", 
	"workspace/sample-project-1/sample-experiment-1/output/models/resume_state-0000.params"]);
#######################################################################################################################




################################################ Reload Model ##########################################################
gtf.Reload();
#######################################################################################################################





######################################## Switch To Eval Mode ########################################
gtf.Switch_Mode(eval_infer=True)
#####################################################################################################




#####################################Evaluate on eval dataset ####################################
gtf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval");
gtf.Dataset();
accuracy, class_based_accuracy = gtf.Evaluate();
#####################################################################################################




######################################## Switch To Train Mode ########################################
gtf.Switch_Mode(train=True)
#####################################################################################################


##################################### Train ###########################################
gtf.update_dataset(dataset_path=["../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
						"../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval"]);

gtf.Dataset(); #Can also use gtf.Reload() here

gtf.Train();
#####################################################################################################




######################################## Switch To Eval Mode ########################################
gtf.Switch_Mode(eval_infer=True)
#####################################################################################################



#####################################Evaluate on eval dataset ####################################
gtf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval");
gtf.Dataset();
accuracy, class_based_accuracy = gtf.Evaluate();
#####################################################################################################