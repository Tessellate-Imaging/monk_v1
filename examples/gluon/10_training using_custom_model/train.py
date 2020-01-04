import os
import sys
sys.path.append("../../../monk/");
import psutil

from gluon_prototype import prototype



gtf = prototype(verbose=1);
gtf.Prototype("sample-project-1", "sample-experiment-1");



######################################################Dataset Params #################################################################
gtf.Dataset_Params(dataset_path="../../../monk/system_check_tests/datasets/dataset_cats_dogs_train", split=0.9,
        input_size=224, batch_size=16, shuffle_data=True, num_processors=3);
#################################################################################################################################################








########################################################### Transforms ####################################################
gtf.apply_random_horizontal_flip(train=True, val=True);
gtf.apply_normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=True, val=True, test=True);
#################################################################################################################################################




########################################################## Set Dataset ###################################################################
gtf.Dataset();
##########################################################################################################################################





############################################ Auxiliary Functions - List all available layers #########################################
gtf.List_Layers_Custom_Model();
######################################################################################################################################



############################################ Auxiliary Functions - List all available activations #########################################
gtf.List_Activations_Custom_Model();
######################################################################################################################################



########################################################## Custom Network ###################################################################
network = [];
network.append(gtf.convolution(output_channels=16));
network.append(gtf.batch_normalization());
network.append(gtf.relu());
network.append(gtf.convolution(output_channels=16));
network.append(gtf.batch_normalization());
network.append(gtf.relu());
network.append(gtf.max_pooling());



subnetwork = [];
branch1 = [];
branch1.append(gtf.convolution(output_channels=16));
branch1.append(gtf.batch_normalization());
branch1.append(gtf.convolution(output_channels=16));
branch1.append(gtf.batch_normalization());

branch2 = [];
branch2.append(gtf.convolution(output_channels=16));
branch2.append(gtf.batch_normalization());

branch3 = [];
branch3.append(gtf.identity())

subnetwork.append(branch1);
subnetwork.append(branch2);
subnetwork.append(branch3);
subnetwork.append(gtf.concatenate());


network.append(subnetwork);



network.append(gtf.convolution(output_channels=16));
network.append(gtf.batch_normalization());
network.append(gtf.relu());
network.append(gtf.max_pooling());


subnetwork = [];
branch1 = [];
branch1.append(gtf.convolution(output_channels=16));
branch1.append(gtf.batch_normalization());
branch1.append(gtf.convolution(output_channels=16));
branch1.append(gtf.batch_normalization());

branch2 = [];
branch2.append(gtf.convolution(output_channels=16));
branch2.append(gtf.batch_normalization());

branch3 = [];
branch3.append(gtf.identity())

subnetwork.append(branch1);
subnetwork.append(branch2);
subnetwork.append(branch3);
subnetwork.append(gtf.add());


network.append(subnetwork);

network.append(gtf.convolution(output_channels=16));
network.append(gtf.batch_normalization());
network.append(gtf.relu());
network.append(gtf.max_pooling());




network.append(gtf.flatten());
network.append(gtf.fully_connected(units=1024));
network.append(gtf.dropout(drop_probability=0.2));
network.append(gtf.fully_connected(units=2));

gtf.debug_custom_model_design(network);
gtf.Compile_Network(network);

##########################################################################################################################################










############################################ Auxiliary Functions - Freeze Layers #########################################
gtf.Freeze_Layers(num=10);
######################################################################################################################################





########################################################## Training Params ###################################################################
gtf.Training_Params(num_epochs=2, display_progress=True, display_progress_realtime=True, 
        save_intermediate_models=True, intermediate_model_prefix="intermediate_model_", save_training_logs=True);
######################################################################################################################################





################################################ Set Optimizer #########################################################################
gtf.optimizer_sgd(0.001);
#################################################################################################################################################


############################################ Auxiliary Functions - List all available optimizers #########################################
gtf.List_Optimizers();
######################################################################################################################################







################################################ Set Learning rate schedulers #################################################################
gtf.lr_fixed();
#################################################################################################################################################


############################################ Auxiliary Functions - List all available schedulers #########################################
gtf.List_Schedulers();
######################################################################################################################################








################################################ Set Loss #################################################################
gtf.loss_softmax_crossentropy()
#################################################################################################################################################


############################################ Auxiliary Functions - List all available losses #########################################
gtf.List_Losses();
######################################################################################################################################




gtf.Train();