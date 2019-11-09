import os
import sys
sys.path.append("../../../monk/");
import psutil

from gluon_prototype import prototype









################################################### Foldered - Train Dataset #################################################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True);

gtf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval");

gtf.Dataset();


accuracy, class_based_accuracy = gtf.Evaluate();
###############################################################################################################################################






######################################################### CSV - Train Dataset #################################################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True);

gtf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_csv_id/train", 
				path_to_csv="../../../monk/system_check_tests/datasets/dataset_csv_id/train.csv");

gtf.Dataset();


accuracy, class_based_accuracy = gtf.Evaluate();
###############################################################################################################################################



