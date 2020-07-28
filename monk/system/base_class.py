from monk.system.imports import *
from monk.system.base_system_state import get_base_system_dict
from monk.system.common import read_json
from monk.system.common import save
from monk.system.common import create_dir
from monk.system.common import delete_dir



class system():
    '''
    Base class for all system project management

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    @accepts("self", verbose=int, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def __init__(self, verbose=1):
        self.system_dict = get_base_system_dict();
        self.system_dict["verbose"] = verbose;
        self.system_dict["cwd"] = os.getcwd() + "/";
        self.system_dict["master_systems_dir"] = self.system_dict["cwd"] + "workspace/";
        self.system_dict["master_systems_dir_relative"] = "workspace/";

        create_dir(self.system_dict["master_systems_dir_relative"]);

        self.system_dict["master_comparison_dir"] = self.system_dict["cwd"] + "workspace/comparison/";
        self.system_dict["master_comparison_dir_relative"] = "workspace/comparison/";
        
        create_dir(self.system_dict["master_comparison_dir_relative"]);
            
        self.system_dict["local"]["projects_list"] = os.listdir(self.system_dict["master_comparison_dir"]);
        self.system_dict["local"]["num_projects"] = len(self.system_dict["local"]["projects_list"]);
        self.system_dict["local"]["experiments_list"] = [];
        self.system_dict["local"]["num_experiments"] = 0;
        self.system_dict["origin"] = ["New", "New"];



    #############################################################################################################################
    @accepts("self", str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_system_project(self, project_name):
        '''
        Create Project

        Args:
            project_name (str): Unique name to project

        Returns:
            None
        '''
        self.system_dict["project_dir"] = self.system_dict["master_systems_dir"] + project_name + "/";
        self.system_dict["project_dir_relative"] = self.system_dict["master_systems_dir_relative"] + project_name + "/";
        if(not os.path.isdir(self.system_dict["project_dir"])):
            self.system_dict["local"]["projects_list"].append(project_name);
            self.system_dict["local"]["num_projects"] += 1;
        create_dir(self.system_dict["project_dir"]);
        self.set_system_select_project(project_name);
    
    
    @accepts("self", str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_system_select_project(self, project_name):
        '''
        Function to update system dictionary on project properties

        Args:
            project_name (str): Unique name to project

        Returns:
            None
        '''
        self.system_dict["project_name"] = project_name;
        self.system_dict["local"]["experiments_list"] = os.listdir(self.system_dict["project_dir"]);
        self.system_dict["local"]["num_experiments"] = len(self.system_dict["local"]["experiments_list"]);


    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_system_aux_list_projects(self):
        '''
        List all projects in current workspace

        Args:
            None

        Returns:
            None
        '''
        return os.listdir(self.system_dict["master_systems_dir"]);
    #############################################################################################################################



    #############################################################################################################################
    @accepts("self", str, eval_infer=bool, copy_from=[list, bool], pseudo_copy_from=[list, bool], resume_train=bool, summary=bool, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_system_experiment(self, experiment_name, eval_infer=False, copy_from=False, pseudo_copy_from=False, resume_train=False, summary=False):
        '''
        Create Experiment or load it in different states

        Args:
            experiment_name (str): Unique name to experiment
            eval_infer (bool): If set as True, model is loaded in evaluation mode
            resume_train (bool): If set as True, model is loaded from last checkpoint
            copy_from (list): [project, experiment] to copy from
            pseudo_copy_from (list): For creating sub-experiments while in hyper-parametric analysis state
            summary (list): Dummy variable

        Returns:
            None
        '''
        if(summary):
            self.set_system_select_experiment(experiment_name);
            print_summary(self.system_dict["fname_relative"]);

        else:
            self.system_dict["experiment_dir"] = self.system_dict["project_dir"] + experiment_name + "/";
            self.system_dict["experiment_dir_relative"] = self.system_dict["project_dir_relative"] + experiment_name + "/";
            if(not os.path.isdir(self.system_dict["experiment_dir"])):
                self.system_dict["local"]["experiments_list"].append(experiment_name);
                self.system_dict["local"]["num_experiments"] += 1;
            create_dir(self.system_dict["experiment_dir"]);
            self.set_system_select_experiment(experiment_name);
            
            if(eval_infer):
                self.set_system_state_eval_infer();
            elif(resume_train):
                self.set_system_state_resume_train();
            elif(copy_from):
                self.set_system_delete_create_dir();
                self.set_system_state_copy_from(copy_from);
            elif(pseudo_copy_from):
                self.set_system_delete_create_dir();
                self.set_system_state_pseudo_copy_from(pseudo_copy_from);
            else: 
                self.set_system_delete_create_dir();
                save(self.system_dict);

    
    @accepts("self", str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_system_select_experiment(self, experiment_name):
        '''
        Function to update system dictionary on experiment properties

        Args:
            experiment_name (str): Unique name to experiment

        Returns:
            None
        '''
        self.system_dict["experiment_name"] = experiment_name;
        self.system_dict["output_dir"] = self.system_dict["experiment_dir"] + "output/";
        self.system_dict["output_dir_relative"] = self.system_dict["experiment_dir_relative"] + "output/";
        self.system_dict["model_dir"] = self.system_dict["output_dir"] + "models/";
        self.system_dict["model_dir_relative"] = self.system_dict["output_dir_relative"] + "models/";
        self.system_dict["log_dir"] = self.system_dict["output_dir"] + "logs/";
        self.system_dict["log_dir_relative"] = self.system_dict["output_dir_relative"] + "logs/";
        self.system_dict["fname"] = self.system_dict["experiment_dir"] + "/experiment_state.json";
        self.system_dict["fname_relative"] = self.system_dict["experiment_dir_relative"] + "/experiment_state.json";
        self.system_dict["visualization"]["base"] = self.system_dict["experiment_dir"] + "visualization/";
        self.system_dict["visualization"]["kernels_dir"] = self.system_dict["visualization"]["base"] + "kernels/";
        self.system_dict["visualization"]["feature_maps_dir"] = self.system_dict["visualization"]["base"] + "feature_maps/";

    #############################################################################################################################



    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_system_delete_create_dir(self):
        '''
        Function to remove old directories and create new at the same place

        Args:
            None

        Returns:
            None
        '''
        delete_dir(self.system_dict["output_dir_relative"]);
        create_dir(self.system_dict["output_dir_relative"]);
        create_dir(self.system_dict["model_dir_relative"]);
        create_dir(self.system_dict["log_dir_relative"]);
    #############################################################################################################################




    #############################################################################################################################
    @accepts("self", str, post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def set_system_comparison(self, comparison_name):
        '''
        Create comparison experiment

        Args:
            comparison_name (str): Unique name to comparison experiment

        Returns:
            None
        '''
        create_dir(self.system_dict["master_comparison_dir"] + comparison_name + "/");
        self.system_dict["comparison_name"] = comparison_name;
        self.system_dict["comparison_dir"] = self.system_dict["master_comparison_dir"] + comparison_name + "/";
    #############################################################################################################################


    #############################################################################################################################
    @accepts("self", [str, int, list, dict, float, tuple], post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def custom_print(self, msg):
        '''
        Overwritten print function, to switch off and on as per verbosity levels

        Args:
            msg (str): Message to print

        Returns:
            None
        '''
        if(self.system_dict["verbose"]):
            print(msg);
    #############################################################################################################################






    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def print_list_models(self):
        '''
        List all the available models in the selected backend.

        Args:
            None

        Returns:
            None
        '''
        self.custom_print("Models List: ");

        if(self.system_dict["library"] == "Mxnet"):
            set1 = ["alexnet", "darknet53", "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201", "InceptionV3", "MobileNet1.0", "MobileNet0.75", 
                        "MobileNet0.25", "MobileNet0.5", "ResNet18_v1", "ResNet34_v1", "ResNet50_v1", "ResNet101_v1", "ResNet152_v1", "ResNext50_32x4d", 
                        "ResNext101_32x4d", "ResNext101_64x4d_v1", "SE_ResNext50_32x4d", "SE_ResNext101_32x4d", "SE_ResNext101_64x4d", "SENet_154", 
                        "VGG11", "VGG13", "VGG16", "VGG19", "VGG11_bn", "VGG13_bn", "VGG16_bn", "VGG19_bn", "ResNet18_v2", "ResNet34_v2", 
                        "ResNet50_v2", "ResNet101_v2", "ResNet152_v2"];
            set2 = ["MobileNetV2_1.0", "MobileNetV2_0.75", "MobileNetV2_0.5", "MobileNetV2_0.25", "SqueezeNet1.0", "SqueezeNet1.1", "MobileNetV3_Large", "MobileNetV3_Small"];
            set3 = ["ResNet18_v1b", "ResNet34_v1b", "ResNet50_v1b", "ResNet50_v1b_gn", "ResNet101_v1b", "ResNet152_v1b", "ResNet50_v1c", 
                        "ResNet101_v1c", "ResNet152_v1c", "ResNet50_v1d", "ResNet101_v1d", "ResNet152_v1d", "ResNet18_v1d", "ResNet34_v1d", 
                        "ResNet50_v1d", "ResNet101_v1d", "ResNet152_v1d", "resnet18_v1b_0.89", "resnet50_v1d_0.86", "resnet50_v1d_0.48", 
                        "resnet50_v1d_0.37", "resnet50_v1d_0.11", "resnet101_v1d_0.76", "resnet101_v1d_0.73", "Xception"];
            combined_list = set1+set2+set3
            combined_list_lower = list(map(str.lower, combined_list))


        elif(self.system_dict["library"] == "Pytorch"):
            set1 = ["alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]
            set2 = ["densenet121", "densenet161", "densenet169", "densenet201"]
            set3 = ["googlenet", "inception_v3", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d",
                        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0, shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "wide_resnet101_2", "wide_resnet50_2"]
            set4 = ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3", "mobilenet_v2", "squeezenet1_0", "squeezenet1_1"]
            combined_list = set1+set2+set3+set4
            combined_list_lower = list(map(str.lower, combined_list))


        elif(self.system_dict["library"] == "Keras"):
            set1 = ["mobilenet", "densenet121", "densenet169", "densenet201", "inception_v3", 
                        "inception_resnet_v3", "mobilenet_v2", "nasnet_mobile", "nasnet_large", "resnet50",
                        "resnet101", "resnet152", "resnet50_v2", "resnet101_v2", "resnet152_v2", "vgg16",
                        "vgg19", "xception"];
            combined_list = set1
            combined_list_lower = list(map(str.lower, combined_list))
            
        for i in range(len(combined_list_lower)):
            self.custom_print("    {}. {}".format(i+1, combined_list_lower[i]))
        self.custom_print("")
    #############################################################################################################################




    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def print_list_layers_transfer_learning(self):
        '''
        List all the available transfer learning layers in the selected backend.

        Args:
            None

        Returns:
            None
        '''
        self.custom_print("Layers List for transfer learning: ");

        if(self.system_dict["library"] == "Mxnet"):
            combined_list_lower = ["append_linear", "append_dropout"];

        elif(self.system_dict["library"] == "Keras"):
            combined_list_lower = ["append_linear", "append_dropout"];

        elif(self.system_dict["library"] == "Pytorch"):
            combined_list_lower = ["append_linear", "append_dropout"];

        for i in range(len(combined_list_lower)):
            self.custom_print("    {}. {}".format(i+1, combined_list_lower[i]))

        self.custom_print("")
    #############################################################################################################################




    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def print_list_layers_custom_model(self):
        '''
        List all the available custom network layers in the selected backend.

        Args:
            None

        Returns:
            None
        '''
        self.custom_print("Layers List for transfer learning: ");

        if(self.system_dict["library"] == "Mxnet"):
            combined_list_lower = ["convolution1d", "convolution2d", "convolution", "convolution3d", "transposed_convolution1d",
                                    "transposed_convolution", "transposed_convolution2d", "transposed_convolution3d", 
                                    "max_pooling1d", "max_pooling2d", "max_pooling", "max_pooling3d", "average_pooling1d",
                                    "average_pooling2d", "average_pooling", "average_pooling3d", "global_max_pooling1d",
                                    "global_max_pooling2d", "global_max_pooling", "global_max_pooling3d", "global_average_pooling1d",
                                    "global_average_pooling2d", "global_average_pooling", "global_average_pooling3d", 
                                    "fully_connected", "dropout", "flatten", "identity", "add", "concatenate", "batch_normalization",
                                    "instance_normalization", "layer_normalization"];

        elif(self.system_dict["library"] == "Keras"):
            combined_list_lower = ["convolution1d", "convolution2d", "convolution", "convolution3d", "transposed_convolution", 
                                    "transposed_convolution2d", "transposed_convolution3d", "max_pooling1d", "max_pooling2d", 
                                    "max_pooling", "max_pooling3d", "average_pooling1d", "average_pooling2d", "average_pooling", 
                                    "average_pooling3d", "global_max_pooling1d", "global_max_pooling2d", "global_max_pooling", 
                                    "global_max_pooling3d", "global_average_pooling1d", "global_average_pooling2d", "global_average_pooling", 
                                    "global_average_pooling3d", "flatten", "fully_connected", "dropout", "identity", "batch_normalization",
                                    "add", "concatenate"];

        elif(self.system_dict["library"] == "Pytorch"):
            combined_list_lower = ["convolution1d", "convolution2d", "convolution", "convolution3d", "transposed_convolution1d",
                                    "transposed_convolution", "transposed_convolution2d", "transposed_convolution3d",
                                    "max_pooling1d", "max_pooling2d", "max_pooling", "max_pooling3d", "average_pooling1d",
                                    "average_pooling2d", "average_pooling", "average_pooling3d", "global_max_pooling1d",
                                    "global_max_pooling2d", "global_max_pooling", "global_max_pooling3d", "global_average_pooling1d",
                                    "global_average_pooling2d", "global_average_pooling", "global_average_pooling3d", "fully_connected", 
                                    "flatten", "dropout", "identity", "batch_normalization", "instance_normalization", "layer_normalization",
                                    "add", "concatenate"];

        for i in range(len(combined_list_lower)):
            self.custom_print("    {}. {}".format(i+1, combined_list_lower[i]))

        self.custom_print("")
    #############################################################################################################################





    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def print_list_activations_transfer_learning(self):
        '''
        List all the available transfer learning activations in the selected backend.

        Args:
            None

        Returns:
            None
        '''
        self.custom_print("Activations List for transfer learning: ");

        if(self.system_dict["library"] == "Mxnet"):
            combined_list_lower = ["append_elu", "append_leakyrelu", "append_prelu", "append_relu", "append_selu",
                                    "append_selu", "append_sigmoid", "append_softplus", "append_tanh",
                                    "append_softmax", "append_swish"];

        elif(self.system_dict["library"] == "Keras"):
            combined_list_lower = ["append_elu", "append_leakyrelu", "append_prelu", "append_relu", "append_selu",
                                    "append_selu", "append_sigmoid", "append_softplus", "append_softsign", "append_tanh",
                                    "append_threshold", "append_softmax"];



        elif(self.system_dict["library"] == "Pytorch"):
            combined_list_lower = ["append_elu", "append_leakyrelu", "append_prelu", "append_relu", "append_selu",
                                    "append_selu", "append_sigmoid", "append_softplus", "append_softsign", "append_tanh",
                                    "append_threshold", "append_softmax", "append_hardshrink", "append_hardtanh", 
                                    "append_logsigmoid", "append_relu6", "append_rrelu", "append_celu", "append_softshrink",
                                    "append_tanhshrink", "append_logsoftmax", "append_softmin"];


        for i in range(len(combined_list_lower)):
            self.custom_print("    {}. {}".format(i+1, combined_list_lower[i]))

        self.custom_print("")
    #############################################################################################################################




    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def print_list_activations_custom_model(self):
        '''
        List all the available custom network activations in the selected backend.

        Args:
            None

        Returns:
            None
        '''
        self.custom_print("Activations List for transfer learning: ");

        if(self.system_dict["library"] == "Mxnet"):
            combined_list_lower = ["relu", "sigmoid", "tanh", "softplus", "softsign", "elu", "gelu", "leaky_relu",
                                    "prelu", "selu", "swish"];

        elif(self.system_dict["library"] == "Keras"):
            combined_list_lower = ["relu", "elu", "leaky_relu", "prelu", "thresholded_relu", "softmax", 
                                    "selu", "softplus", "softsign", "tanh", "sigmoid", "hard_sigmoid"];



        elif(self.system_dict["library"] == "Pytorch"):
            combined_list_lower = ["relu", "sigmoid", "tanh", "softplus", "softsign",  "elu", "leaky_relu", "prelu", "selu",
                                    "hardshrink", "hardtanh", "logsigmoid", "relu6", "rrelu", "celu", "softshrink", "tanhshrink",
                                    "threshold", "softmin", "softmax", "logsoftmax"];


        for i in range(len(combined_list_lower)):
            self.custom_print("    {}. {}".format(i+1, combined_list_lower[i]))

        self.custom_print("")
    #############################################################################################################################






    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def print_list_losses(self):
        '''
        List all the available loss functions in the selected backend.

        Args:
            None

        Returns:
            None
        '''
        self.custom_print("Losses List: ");

        if(self.system_dict["library"] == "Mxnet"):
            combined_list_lower = ["loss_l1", "loss_l2", "loss_softmax_crossentropy", "loss_crossentropy",
                                    "loss_sigmoid_binary_crossentropy", "loss_binary_crossentropy",
                                    "loss_kldiv", "loss_poisson_nll", "loss_huber", "loss_hinge",
                                    "loss_squared_hinge"];

        elif(self.system_dict["library"] == "Keras"):
            combined_list_lower = ["loss_l1", "loss_l2", "loss_crossentropy", "loss_binary_crossentropy", 
                                    "loss_kldiv", "loss_hinge", "loss_squared_hinge"];


        elif(self.system_dict["library"] == "Pytorch"):
            combined_list_lower = ["loss_l1", "loss_l2", "loss_softmax_crossentropy", "loss_crossentropy",
                                    "loss_sigmoid_binary_crossentropy", "loss_binary_crossentropy",
                                    "loss_kldiv", "loss_poisson_nll", "loss_huber", "loss_hinge",
                                    "loss_squared_hinge", "loss_multimargin", "loss_squared_multimargin",
                                    "loss_multilabel_margin", "loss_multilabel_softmargin"];

                                    

        for i in range(len(combined_list_lower)):
            self.custom_print("    {}. {}".format(i+1, combined_list_lower[i]))

        self.custom_print("")
    #############################################################################################################################





    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def print_list_optimizers(self):
        '''
        List all the available optimizers in the selected backend.

        Args:
            None

        Returns:
            None
        '''
        self.custom_print("Optimizers List: ");

        if(self.system_dict["library"] == "Mxnet"):
            combined_list_lower = ["optimizer_sgd", "optimizer_nesterov_sgd", "optimizer_rmsprop", "optimizer_momentum_rmsprop", 
                                    "optimizer_adam", "optimizer_adagrad", "optimizer_nesterov_adam", 
                                    "optimizer_adadelta", "optimizer_adamax", "optimizer_signum"];

        elif(self.system_dict["library"] == "Keras"):
            combined_list_lower = ["optimizer_sgd", "optimizer_nesterov_sgd", "optimizer_rmsprop", "optimizer_adam",
                                    "optimizer_nesterov_adam", "optimizer_adamax", "optimizer_adagrad", "optimizer_adadelta"];


        elif(self.system_dict["library"] == "Pytorch"):
            combined_list_lower = ["optimizer_sgd", "optimizer_nesterov_sgd", "optimizer_rmsprop", "optimizer_momentum_rmsprop", 
                                    "optimizer_adam", "optimizer_adamax", "optimizer_adamw", "optimizer_adagrad", 
                                    "optimizer_adadelta"];

                                    
        for i in range(len(combined_list_lower)):
            self.custom_print("    {}. {}".format(i+1, combined_list_lower[i]))

        self.custom_print("")
    #############################################################################################################################





    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def print_list_schedulers(self):
        '''
        List all the available learning rate schedulers in the selected backend.

        Args:
            None

        Returns:
            None
        '''
        self.custom_print("Optimizers List: ");

        if(self.system_dict["library"] == "Mxnet"):
            combined_list_lower = ["lr_fixed", "lr_step_decrease", "lr_multistep_decrease"];

        elif(self.system_dict["library"] == "Keras"):
            combined_list_lower = ["lr_fixed", "lr_step_decrease", "lr_exponential_decrease", "lr_plateau_decrease"];


        elif(self.system_dict["library"] == "Pytorch"):
            combined_list_lower = ["lr_fixed", "lr_step_decrease", "lr_multistep_decrease", "lr_exponential_decrease", "lr_plateau_decrease"];

                                    
        for i in range(len(combined_list_lower)):
            self.custom_print("    {}. {}".format(i+1, combined_list_lower[i]))

        self.custom_print("")
    #############################################################################################################################






    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def print_list_transforms(self):
        '''
        List all the available data transforms in the selected backend.

        Args:
            None

        Returns:
            None
        '''
        self.custom_print("Transforms List: ");

        if(self.system_dict["library"] == "Mxnet"):
            combined_list_lower = ["apply_random_resized_crop", "apply_center_crop", "apply_color_jitter", "apply_random_horizontal_flip",
                                    "apply_random_vertical_flip", "apply_random_lighting", "apply_resize", "apply_normalize"];

        elif(self.system_dict["library"] == "Keras"):
            combined_list_lower = ["apply_color_jitter", "apply_random_affine", "apply_random_horizontal_flip", 
                                    "apply_random_vertical_flip", "apply_random_rotation", "apply_mean_subtraction", 
                                    "apply_normalize"];


        elif(self.system_dict["library"] == "Pytorch"):
            combined_list_lower = ["apply_center_crop", "apply_color_jitter", "apply_random_affine", "apply_random_crop", 
                                    "apply_random_horizontal_flip", "apply_random_perspective", "apply_random_resized_crop",
                                    "apply_grayscale", "apply_random_rotation", "apply_random_vertical_flip",
                                    "apply_resize", "apply_normalize"];

                                    
        for i in range(len(combined_list_lower)):
            self.custom_print("    {}. {}".format(i+1, combined_list_lower[i]))

        self.custom_print("")
    #############################################################################################################################





    #############################################################################################################################
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=True, trace_rv=True)
    def print_list_blocks(self):
        '''
        List all the available blocks for custom network creation in the selected backend.

        Args:
            None

        Returns:
            None
        '''
        self.custom_print("Blocks List: ");

        if(self.system_dict["library"] == "Mxnet"):
            combined_list_lower = ["resnet_v1_block", "resnet_v2_block", "resnet_v1_bottleneck_block", "resnet_v2_bottleneck_block",
                                    "resnext_block", "mobilenet_v2_linear_block", "mobilenet_v2_inverted_linear_block",
                                    "squeezenet_fire_block", "conv_bn_relu_block", "inception_a_block", "inception_b_block",
                                    "inception_c_block", "inception_d_block", "inception_e_block"];

        elif(self.system_dict["library"] == "Keras"):
            combined_list_lower = ["resnet_v1_block", "resnet_v2_block", "resnet_v1_bottleneck_block", "resnet_v2_bottleneck_block",
                                    "resnext_block", "mobilenet_v2_linear_block", "mobilenet_v2_inverted_linear_block",
                                    "squeezenet_fire_block", "conv_bn_relu_block", "inception_a_block", "inception_b_block",
                                    "inception_c_block", "inception_d_block", "inception_e_block"];


        elif(self.system_dict["library"] == "Pytorch"):
            combined_list_lower = ["resnet_v1_block", "resnet_v2_block", "resnet_v1_bottleneck_block", "resnet_v2_bottleneck_block",
                                    "resnext_block", "mobilenet_v2_linear_block", "mobilenet_v2_inverted_linear_block",
                                    "squeezenet_fire_block", "conv_bn_relu_block", "inception_a_block", "inception_b_block",
                                    "inception_c_block", "inception_d_block", "inception_e_block"];

                                    
        for i in range(len(combined_list_lower)):
            self.custom_print("    {}. {}".format(i+1, combined_list_lower[i]))

        self.custom_print("")
    #############################################################################################################################