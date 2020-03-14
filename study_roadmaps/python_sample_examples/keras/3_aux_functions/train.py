import os
import sys
sys.path.append("../../../monk/");
import psutil

from keras_prototype import prototype



ktf = prototype(verbose=1);
ktf.Prototype("sample-project-1", "sample-experiment-1");


ktf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet50", freeze_base_network=True, num_epochs=2);




########################################################   Summary    #####################################################
ktf.Summary()
###########################################################################################################################





##################################################### EDA - Find Num images per class #####################################
ktf.EDA(show_img=True, save_img=True);
###########################################################################################################################





##################################################### EDA - Find Missing and corrupted images #####################################
ktf.EDA(check_missing=True, check_corrupt=True);
###########################################################################################################################





##################################################### Estimate Training Time #####################################
ktf.Estimate_Train_Time(num_epochs=50);
###########################################################################################################################