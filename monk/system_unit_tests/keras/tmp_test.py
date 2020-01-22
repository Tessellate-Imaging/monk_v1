import os
import sys

sys.path.append("../../../monk/");
import psutil

from keras_prototype import prototype




gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");
gtf.Default(dataset_path="../../system_check_tests/datasets/dataset_cats_dogs_train", 
    model_name="resnet50", freeze_base_network=True, num_epochs=2);
gtf.optimizer_nesterov_adam(0.01, weight_decay=0.0001, beta1=0.9, beta2=0.999, 
            	clipnorm=1.0, clipvalue=0.5);
gtf.Train();