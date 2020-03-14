import os
import sys
sys.path.append("../../../monk/");
import psutil

from pytorch_prototype import prototype









################################################### Foldered - Train Dataset #################################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True);

ptf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval");

ptf.Dataset();


accuracy, class_based_accuracy = ptf.Evaluate();
###############################################################################################################################################






######################################################### CSV - Train Dataset #################################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True);

ptf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_csv_id/train", 
				path_to_csv="../../../monk/system_check_tests/datasets/dataset_csv_id/train.csv");

ptf.Dataset();


accuracy, class_based_accuracy = ptf.Evaluate();
###############################################################################################################################################



