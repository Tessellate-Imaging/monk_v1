import os
import sys
sys.path.append("../../../monk/");
import psutil

from pytorch_prototype import prototype



ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-2");


ptf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet18", freeze_base_network=True, num_epochs=2);




##################################################  Reset Model #######################################################
ptf.reset_model();
#######################################################################################################################



############################################## Load model from external path ##########################################
ptf.update_model_path("workspace/sample-project-1/sample-experiment-1/output/models/best_model");
#######################################################################################################################




################################################ Reload Model ##########################################################
ptf.Reload();
#######################################################################################################################





######################################## Switch To Eval Mode ########################################
ptf.Switch_Mode(eval_infer=True)
#####################################################################################################




#####################################Evaluate on eval dataset ####################################
ptf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval");
ptf.Dataset();
accuracy, class_based_accuracy = ptf.Evaluate();
#####################################################################################################




######################################## Switch To Train Mode ########################################
ptf.Switch_Mode(train=True)
#####################################################################################################


##################################### Train ###########################################
ptf.update_dataset(dataset_path=["../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
						"../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval"]);

ptf.Dataset(); #Can also use ptf.Reload() here

ptf.Train();
#####################################################################################################




######################################## Switch To Eval Mode ########################################
ptf.Switch_Mode(eval_infer=True)
#####################################################################################################



#####################################Evaluate on eval dataset ####################################
ptf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval");
ptf.Dataset();
accuracy, class_based_accuracy = ptf.Evaluate();
#####################################################################################################