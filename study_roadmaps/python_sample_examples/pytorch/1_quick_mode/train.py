import os
import sys
sys.path.append("../../../monk/");
import psutil

from pytorch_prototype import prototype





################################################## Foldered - Train Dataset #########################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1");


ptf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet18", freeze_base_network=True, num_epochs=2);


ptf.Train();

######################################################################################################################################





############################################ Auxiliary Functions - List all available models #########################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1");

ptf.List_Models();
######################################################################################################################################







################################################## Foldered - Train and Val Dataset ###################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1");


ptf.Default(dataset_path=["../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
							"../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval"], 
    			model_name="resnet18", freeze_base_network=True, num_epochs=2);


ptf.Train();

######################################################################################################################################





################################################## CSV - Train Dataset ##############################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1");


ptf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_csv_id/train", 
				path_to_csv="../../../monk/system_check_tests/datasets/dataset_csv_id/train.csv",
    			model_name="resnet18", freeze_base_network=True, num_epochs=2);


ptf.Train();

######################################################################################################################################






################################################## CSV - Train and Val Dataset ##############################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1");


ptf.Default(dataset_path=["../../../monk/system_check_tests/datasets/dataset_csv_id/train", 
				"../../../monk/system_check_tests/datasets/dataset_csv_id/val"],
				path_to_csv=["../../../monk/system_check_tests/datasets/dataset_csv_id/train.csv",
					"../../../monk/system_check_tests/datasets/dataset_csv_id/val.csv"],
    			model_name="resnet18", freeze_base_network=True, num_epochs=2);


ptf.Train();

######################################################################################################################################
