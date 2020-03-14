import os
import sys
sys.path.append("../../../monk/");
import psutil

from keras_prototype import prototype









################################################### Foldered - Train Dataset #################################################################
ktf = prototype(verbose=1);
ktf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True);

ktf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_eval");

ktf.Dataset();


accuracy, class_based_accuracy = ktf.Evaluate();
###############################################################################################################################################






######################################################### CSV - Train Dataset #################################################################
ktf = prototype(verbose=1);
ktf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True);

ktf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_csv_id/train", 
				path_to_csv="../../../monk/system_check_tests/datasets/dataset_csv_id/train.csv");

ktf.Dataset();


accuracy, class_based_accuracy = ktf.Evaluate();
###############################################################################################################################################



