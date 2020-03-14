import os
import sys
sys.path.append("../../../monk/");
import psutil

from pytorch_prototype import prototype


###################################################################################################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-1");


ptf.Default(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", 
    			model_name="resnet18", freeze_base_network=True, num_epochs=2);

ptf.Train();
###################################################################################################################





########################################### Continue from experiment - 1 ##############################################
ptf = prototype(verbose=1);
ptf.Prototype("sample-project-1", "sample-experiment-2", copy_from=["sample-project-1", "sample-experiment-1"]);



# Reset Transforms if required
ptf.reset_transforms();
ptf.reset_transforms(test=True);
# Apply new transforms
ptf.apply_random_horizontal_flip(train=True, val=True);
ptf.apply_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=True, val=True, test=True);


# See all available transforms
ptf.List_Transforms()


# Make necessary updates
ptf.update_num_epochs(3);
ptf.optimizer_adam(0.001);

# Reload function
ptf.Reload();

#Retrain
ptf.Train();
###################################################################################################################