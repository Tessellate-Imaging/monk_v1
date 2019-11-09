import os
import sys
sys.path.append("../../../monk/");
import psutil

from gluon_prototype import prototype





################################################## Foldered - Train Dataset #########################################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");


gtf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet18_v1", freeze_base_network=True, num_epochs=2);


gtf.Train();

######################################################################################################################################





############################################ Auxiliary Functions - List all available models #########################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");

gtf.List_Models();
######################################################################################################################################







################################################## Foldered - Train and Val Dataset ###################################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");


gtf.Default(dataset_path=["../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
							"../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval"], 
    			model_name="resnet18_v1", freeze_base_network=True, num_epochs=2);


gtf.Train();

######################################################################################################################################





################################################## CSV - Train Dataset ##############################################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");


gtf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_csv_id/train", 
				path_to_csv="../../../monk/system_check_tests/datasets/dataset_csv_id/train.csv",
    			model_name="resnet18_v1", freeze_base_network=True, num_epochs=2);


gtf.Train();

######################################################################################################################################






################################################## CSV - Train and Val Dataset ##############################################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");


gtf.Default(dataset_path=["../../../monk/system_check_tests/datasets/dataset_csv_id/train", 
				"../../../monk/system_check_tests/datasets/dataset_csv_id/val"],
				path_to_csv=["../../../monk/system_check_tests/datasets/dataset_csv_id/train.csv",
					"../../../monk/system_check_tests/datasets/dataset_csv_id/val.csv"],
    			model_name="resnet18_v1", freeze_base_network=True, num_epochs=2);


gtf.Train();

######################################################################################################################################
