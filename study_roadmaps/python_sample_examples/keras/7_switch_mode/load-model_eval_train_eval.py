import os
import sys
sys.path.append("../../../monk/");
import psutil

from keras_prototype import prototype



ktf = prototype(verbose=1);
ktf.Prototype("sample-project-1", "sample-experiment-2");


ktf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet50", freeze_base_network=True, num_epochs=2);




##################################################  Reset Model #######################################################
ktf.reset_model();
#######################################################################################################################



############################################## Load model from external path ##########################################
ktf.update_model_path("workspace/sample-project-1/sample-experiment-1/output/models/best_model.h5");
#######################################################################################################################




################################################ Reload Model ##########################################################
ktf.Reload();
#######################################################################################################################





######################################## Switch To Eval Mode ########################################
ktf.Switch_Mode(eval_infer=True)
#####################################################################################################




#####################################Evaluate on eval dataset ####################################
ktf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval");
ktf.Dataset();
accuracy, class_based_accuracy = ktf.Evaluate();
#####################################################################################################




######################################## Switch To Train Mode ########################################
ktf.Switch_Mode(train=True)
#####################################################################################################


##################################### Train ###########################################
ktf.update_dataset(dataset_path=["../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
						"../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval"]);

ktf.Dataset(); #Can also use ktf.Reload() here

ktf.Train();
#####################################################################################################




######################################## Switch To Eval Mode ########################################
ktf.Switch_Mode(eval_infer=True)
#####################################################################################################



#####################################Evaluate on eval dataset ####################################
ktf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval");
ktf.Dataset();
accuracy, class_based_accuracy = ktf.Evaluate();
#####################################################################################################