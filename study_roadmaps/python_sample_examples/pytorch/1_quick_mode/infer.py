import os
import sys
sys.path.append("../../../monk/");
import psutil

from pytorch_prototype import prototype



###################################################       Inference on single image     #############################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True);


img_name = "../../../monk/system_check_tests/datasets/dataset_cats_dogs_test/0.jpg";
predictions = ptf.Infer(img_name=img_name, return_raw=True);


img_name = "../../../monk/system_check_tests/datasets/dataset_cats_dogs_test/88.jpg";
predictions = ptf.Infer(img_name=img_name, return_raw=True);

#######################################################################################################################################################






###########################################################  Inference on images inside a folder  ##########################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1", eval_infer=True);


inference_dataset = "../../../monk/system_check_tests/datasets/dataset_cats_dogs_test/";
output = ptf.Infer(img_dir=inference_dataset, return_raw=True);
print(output[0:10]);
print("");
#######################################################################################################################################################