import os
import sys

sys.path.append("../../../monk/");
import psutil

from pytorch_prototype import prototype




gtf = prototype(verbose=0);
gtf.Prototype("sample-project-1", "sample-experiment-1");
gtf.Default(dataset_path="../../system_check_tests/datasets/dataset_cats_dogs_train", 
    model_name="resnet18", freeze_base_network=True, num_epochs=2);
gtf.optimizer_adagrad(0.01, weight_decay=0.0001, learning_rate_decay=0.001);
gtf.Train();