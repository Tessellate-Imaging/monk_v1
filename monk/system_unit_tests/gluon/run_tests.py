import os
import sys
import time

from test_optimizer_sgd import test_optimizer_sgd
from test_optimizer_nesterov_sgd import test_optimizer_nesterov_sgd
from test_optimizer_rmsprop import test_optimizer_rmsprop
from test_optimizer_momentum_rmsprop import test_optimizer_momentum_rmsprop
from test_optimizer_adam import test_optimizer_adam
from test_optimizer_adamax import test_optimizer_adamax
from test_optimizer_adadelta import test_optimizer_adadelta
from test_optimizer_adagrad import test_optimizer_adagrad
from test_optimizer_nadam import test_optimizer_nadam
from test_optimizer_signum import test_optimizer_signum

from test_loss_l1 import test_loss_l1
from test_loss_l2 import test_loss_l2
from test_loss_softmax_crossentropy import test_loss_softmax_crossentropy
from test_loss_crossentropy import test_loss_crossentropy
from test_loss_sigmoid_binary_crossentropy import test_loss_sigmoid_binary_crossentropy
from test_loss_binary_crossentropy import test_loss_binary_crossentropy
from test_loss_kldiv import test_loss_kldiv
from test_loss_poisson_nll import test_loss_poisson_nll
from test_loss_huber import test_loss_huber
from test_loss_hinge import test_loss_hinge
from test_loss_squared_hinge import test_loss_squared_hinge


from test_layer_convolution1d import test_layer_convolution1d
from test_layer_convolution2d import test_layer_convolution2d
from test_layer_convolution3d import test_layer_convolution3d
from test_layer_transposed_convolution1d import test_layer_transposed_convolution1d
from test_layer_transposed_convolution2d import test_layer_transposed_convolution2d
from test_layer_transposed_convolution3d import test_layer_transposed_convolution3d
from test_layer_max_pooling1d import test_layer_max_pooling1d
from test_layer_max_pooling2d import test_layer_max_pooling2d
from test_layer_max_pooling3d import test_layer_max_pooling3d
from test_layer_average_pooling1d import test_layer_average_pooling1d
from test_layer_average_pooling2d import test_layer_average_pooling2d
from test_layer_average_pooling3d import test_layer_average_pooling3d
from test_layer_global_max_pooling1d import test_layer_global_max_pooling1d
from test_layer_global_max_pooling2d import test_layer_global_max_pooling2d
from test_layer_global_max_pooling3d import test_layer_global_max_pooling3d
from test_layer_global_average_pooling1d import test_layer_global_average_pooling1d
from test_layer_global_average_pooling2d import test_layer_global_average_pooling2d
from test_layer_global_average_pooling3d import test_layer_global_average_pooling3d
from test_layer_batch_normalization import test_layer_batch_normalization
from test_layer_instance_normalization import test_layer_instance_normalization
from test_layer_layer_normalization import test_layer_layer_normalization
from test_layer_identity import test_layer_identity
from test_layer_fully_connected import test_layer_fully_connected
from test_layer_dropout import test_layer_dropout
from test_layer_flatten import test_layer_flatten
from test_activation_relu import test_activation_relu
from test_activation_sigmoid import test_activation_sigmoid 
from test_activation_tanh import test_activation_tanh
from test_activation_softplus import test_activation_softplus
from test_activation_softsign import test_activation_softsign
from test_activation_elu import test_activation_elu
from test_activation_gelu import test_activation_gelu
from test_activation_prelu import test_activation_prelu
from test_activation_leaky_relu import test_activation_leaky_relu
from test_activation_selu import test_activation_selu
from test_activation_swish import test_activation_swish
from test_layer_concatenate import test_layer_concatenate

from test_initializer_xavier_normal import test_initializer_xavier_normal
from test_initializer_xavier_uniform import test_initializer_xavier_uniform
from test_initializer_orthogonal_normal import test_initializer_orthogonal_normal
from test_initializer_orthogonal_uniform import test_initializer_orthogonal_uniform
from test_initializer_normal import test_initializer_normal
from test_initializer_uniform import test_initializer_uniform
from test_initializer_msra import test_initializer_msra


origstdout = sys.stdout


print("Running Tests...");
sys.stdout = open("test_logs.txt", 'w');


try:
    print("System Check");
    from mxnet.runtime import feature_list
    print("Runtime elements     - {}".format(feature_list()));
    print("");
except:
    print("Installation incomplete");
    sys.exit(0);


system_dict = {};
system_dict["total_tests"] = 0;
system_dict["successful_tests"] = 0;
system_dict["failed_tests_lists"] = [];
system_dict["failed_tests_exceptions"] = [];
system_dict["skipped_tests_lists"] = [];


start = time.time()



print("Running 1/<num>");
system_dict = test_layer_max_pooling1d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 2/<num>");
system_dict = test_layer_max_pooling2d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 3/<num>");
system_dict = test_layer_max_pooling3d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 4/<num>");
system_dict = test_layer_average_pooling1d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 5/<num>");
system_dict = test_layer_average_pooling2d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 6/<num>");
system_dict = test_layer_average_pooling3d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 7/<num>");
system_dict = test_layer_global_max_pooling1d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 8/<num>");
system_dict = test_layer_global_max_pooling2d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 35/<num>");
system_dict = test_layer_global_max_pooling3d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 9/<num>");
system_dict = test_layer_global_average_pooling1d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 10/<num>");
system_dict = test_layer_global_average_pooling2d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 11/<num>");
system_dict = test_layer_global_average_pooling3d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 12/<num>");
system_dict = test_optimizer_sgd(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 12/<num>");
system_dict = test_optimizer_nesterov_sgd(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")

print("Running 14/<num>");
system_dict = test_optimizer_rmsprop(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 15/<num>");
system_dict = test_optimizer_momentum_rmsprop(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 16/<num>");
system_dict = test_optimizer_adam(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 17/<num>");
system_dict = test_optimizer_adamax(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 18/<num>");
system_dict = test_optimizer_adadelta(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 19/<num>");
system_dict = test_optimizer_adagrad(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 20/<num>");
system_dict = test_optimizer_nadam(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 21/<num>");
system_dict = test_optimizer_signum(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")







print("Running 22/<num>");
system_dict = test_loss_l1(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 23/<num>");
system_dict = test_loss_l2(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 24/<num>");
system_dict = test_loss_softmax_crossentropy(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")




print("Running 25/<num>");
system_dict = test_loss_crossentropy(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 26/<num>");
system_dict = test_loss_sigmoid_binary_crossentropy(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 27/<num>");
system_dict = test_loss_binary_crossentropy(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 28/<num>");
system_dict = test_loss_kldiv(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 29/<num>");
system_dict = test_loss_poisson_nll(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 30/<num>");
system_dict = test_loss_huber(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 31/<num>");
system_dict = test_loss_hinge(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 32/<num>");
system_dict = test_loss_squared_hinge(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



















print("Running 33/<num>");
system_dict = test_layer_batch_normalization(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 34/<num>");
system_dict = test_layer_instance_normalization(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 35/<num>");
system_dict = test_layer_layer_normalization(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 36/<num>");
system_dict = test_layer_identity(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 37/<num>");
system_dict = test_layer_fully_connected(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 38/<num>");
system_dict = test_layer_dropout(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 39/<num>");
system_dict = test_layer_flatten(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")





print("Running 40/<num>");
system_dict = test_activation_relu(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 41/<num>");
system_dict = test_activation_sigmoid(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 42/<num>");
system_dict = test_activation_tanh(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 43/<num>");
system_dict = test_activation_softplus(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 44/<num>");
system_dict = test_activation_softsign(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 45/<num>");
system_dict = test_activation_elu(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 46/<num>");
system_dict = test_activation_gelu(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 47/<num>");
system_dict = test_activation_prelu(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")

print("Running 48/<num>");
system_dict = test_activation_leaky_relu(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")

print("Running 49/<num>");
system_dict = test_activation_selu(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 50/<num>");
system_dict = test_activation_swish(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 51/<num>");
system_dict = test_layer_concatenate(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")





print("Running 52/<num>");
system_dict = test_initializer_xavier_normal(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")

print("Running 53/<num>");
system_dict = test_initializer_xavier_uniform(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")

print("Running 54/<num>");
system_dict = test_initializer_orthogonal_normal(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")

print("Running 55/<num>");
system_dict = test_initializer_orthogonal_uniform(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")

print("Running 56/<num>");
system_dict = test_initializer_normal(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")

print("Running 57/<num>");
system_dict = test_initializer_uniform(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")

print("Running 58/<num>");
system_dict = test_initializer_msra(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")




print("Running 59/<num>");
system_dict = test_layer_convolution1d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")



print("Running 60/<num>");
system_dict = test_layer_convolution2d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 61/<num>");
system_dict = test_layer_convolution3d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 62/<num>");
system_dict = test_layer_transposed_convolution1d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 63/<num>");
system_dict = test_layer_transposed_convolution2d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 64/<num>");
system_dict = test_layer_transposed_convolution3d(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")




sys.stdout = open("test_logs.txt", 'a');
end = time.time();

print("Total Tests           - {}".format(system_dict["total_tests"]));
print("Time Taken            - {} sec".format(end-start));
print("Num Successful Tests  - {}".format(system_dict["successful_tests"]));
print("Num Failed Tests      - {}".format(len(system_dict["failed_tests_lists"])));
print("Num Skipped Tests     - {}".format(len(system_dict["skipped_tests_lists"])));
print("");



for i in range(len(system_dict["failed_tests_lists"])):
    print("{}. Failed Test:".format(i+1));
    print("Name     - {}".format(system_dict["failed_tests_lists"][i]));
    print("Error    - {}".format(system_dict["failed_tests_exceptions"][i]));
    print("");


print("Skipped Tests List    - {}".format(system_dict["skipped_tests_lists"]));
print("");


sys.stdout = origstdout;


print("Total Tests           - {}".format(system_dict["total_tests"]));
print("Time Taken            - {} sec".format(end-start));
print("Num Successful Tests  - {}".format(system_dict["successful_tests"]));
print("Num Failed Tests      - {}".format(len(system_dict["failed_tests_lists"])));
print("Num Skipped Tests     - {}".format(len(system_dict["skipped_tests_lists"])));
print("See test_logs.txt for errors");
print("");


os.system("rm -r workspace");