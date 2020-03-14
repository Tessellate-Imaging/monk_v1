import os
import sys
sys.path.append("../../../monk/");
import psutil

from keras_prototype import prototype


###################################################################################################################
ktf = prototype(verbose=1);
ktf.Prototype("sample-project-1", "sample-experiment-1");


ktf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet50", freeze_base_network=True, num_epochs=2);

ktf.Train();
###################################################################################################################





########################################### Continue from experiment - 1 ##############################################
ktf = prototype(verbose=1);
ktf.Prototype("sample-project-1", "sample-experiment-2", copy_from=["sample-project-1", "sample-experiment-1"]);



# Reset Transforms if required
ktf.reset_transforms();
ktf.reset_transforms(test=True);
# Apply new transforms
ktf.apply_random_horizontal_flip(train=True, val=True);
ktf.apply_mean_subtraction(mean=[0.485, 0.456, 0.406], train=True, val=True, test=True);


# See all available transforms
ktf.List_Transforms()


# Make necessary updates
ktf.update_num_epochs(3);
ktf.optimizer_adam(0.0001);

# Reload function
ktf.Reload();

#Retrain
ktf.Train();
###################################################################################################################