import os
import sys
import time

from test_optimizer_sgd import test_optimizer_sgd
from test_optimizer_nesterov_sgd import test_optimizer_nesterov_sgd
from test_optimizer_rmsprop import test_optimizer_rmsprop
from test_optimizer_momentum_rmsprop import test_optimizer_momentum_rmsprop
from test_optimizer_adam import test_optimizer_adam
from test_optimizer_adamax import test_optimizer_adamax
from test_optimizer_adamw import test_optimizer_adamw
from test_optimizer_adadelta import test_optimizer_adadelta
from test_optimizer_adagrad import test_optimizer_adagrad


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
system_dict = test_optimizer_sgd(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 2/<num>");
system_dict = test_optimizer_nesterov_sgd(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 3/<num>");
system_dict = test_optimizer_rmsprop(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 4/<num>");
system_dict = test_optimizer_adam(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 5/<num>");
system_dict = test_optimizer_adamax(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 6/<num>");
system_dict = test_optimizer_adamw(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 7/<num>");
system_dict = test_optimizer_adadelta(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 8/<num>");
system_dict = test_optimizer_adagrad(system_dict)
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