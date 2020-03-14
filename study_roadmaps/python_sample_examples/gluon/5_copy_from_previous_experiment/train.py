import os
import sys
sys.path.append("../../../monk/");
import psutil

from gluon_prototype import prototype


###################################################################################################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");


gtf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet18_v1", freeze_base_network=True, num_epochs=2);

gtf.Train();
###################################################################################################################





########################################### Continue from experiment - 1 ##############################################
gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-2", copy_from=["sample-project-1", "sample-experiment-1"]);



# Reset Transforms if required
gtf.reset_transforms();
gtf.reset_transforms(test=True);
# Apply new transforms
gtf.apply_random_horizontal_flip(train=True, val=True);
gtf.apply_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=True, val=True, test=True);


# See all available transforms
gtf.List_Transforms()


# Make necessary updates
gtf.update_num_epochs(3);
gtf.optimizer_adam(0.001);

# Reload function
gtf.Reload();

#Retrain
gtf.Train();
###################################################################################################################