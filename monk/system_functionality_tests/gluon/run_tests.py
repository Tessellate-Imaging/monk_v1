import os
import sys
import time
from test_default_train import test_default_train
from test_default_eval_infer import test_default_eval_infer
from test_update_copy_from import test_update_copy_from
from test_update_normal import test_update_normal
from test_update_eval_infer import test_update_eval_infer
from test_expert_train import test_expert_train
from test_expert_eval_infer import test_expert_eval_infer
from test_switch_default import test_switch_default
from test_switch_expert import test_switch_expert
from test_compare import test_compare
from test_analyse import test_analyse
origstdout = sys.stdout

#default - train
#   - Object creation
#   - Prototype()
#   - Default() - foldered - train
#   - Train()


#default - eval_infer
#   - Object creation
#   - Prototype()
#   - infer()
#       - img
#       - directory
#   - Dataset_Params() - foldered
#   - Dataset()
#   - Evaluate()


#update - copy_from
#   - Object creation
#   - Prototype()
#   - reset_transforms()
#   - apply_<transforms> - yes
#   - update_dataset_<all>  - foldered - trainval
#   - Reload()
#   - EDA()
#   - Train()


#update - normal
#   - Object creation
#   - Prototype()
#   - Default - csv_train
#   - update_model_<all>
#   - update_training_<all>
#   - lr_<>()
#   - Reload()
#   - EDA()
#   - ETA()
#   - Train()


#update - eval_infer 
#   - Object creation
#   - Prototype()
#   - Dataset_Params() - csv
#   - Dataset()
#   - update_model_path() - external path
#   - Reload()
#   - Evaluate()



#expert - train
#   - Object creation
#   - Prototype()
#   - Dataset_Params() - csv - trainval
#   - apply_<transforms>() - off
#   - Dataset()
#   - Model_Params()
#   - append_<layer>() - off
#   - Model()
#   - Freeze_Layers() - off
#   - lr_<>()
#   - optimizer_<>()
#   - loss_<>()
#   - Training_Params()
#   - Train()


#expert - eval_infer
#   - Object creation
#   - Prototype()
#   - Dataset_Params() - csv
#   - reset_transforms()
#   - apply_<transforms>() - off
#   - Dataset()
#   - Evaluate()


#switch - default
#   - Object creation
#   - Prototype()
#   - Default() - foldered - trainval
#   - EDA()
#   - Switch_Mode() - Eval
#   - Dataset_Params() - foldered
#   - Dataset()
#   - Evaluate()
#   - Switch_Mode() - Train 
#   - Train()




#switch - expert
#   - Object creation
#   - Prototype()
#   - Switch_Mode() - Eval
#   - update_model_path()
#   - update_use_gpu()
#   - Model
#   - update_input_size
#   - Infer - img
#   - Infer - folder
#   - Dataset_Params() - foldered
#   - Dataset()
#   - Evaluate()
#   - Switch_Mode() - Train 
#   - Dataset_Params() - foldered - train
#   - apply_<transforms>() - on
#   - Dataset()
#   - Model_Params()
#   - Model()
#   - lr_<>()
#   - optimizer_<>()
#   - loss_<>()
#   - Training_Params()
#   - Train()
#   - Switch_Mode() - Eval
#   - Dataset_Params() - foldered
#   - Dataset()
#   - Evaluate()


#compare
#   - Object creation
#   - Comparison()
#   - add_experiment()
#   - generate_statistics()

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


print("Running 1/11");
system_dict = test_default_train(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));    
print("")


print("Running 2/11");
sys.stdout = open("test_logs.txt", 'a');
system_dict = test_default_eval_infer(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));
print("")


print("Running 3/11");
sys.stdout = open("test_logs.txt", 'a');
system_dict = test_update_copy_from(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));
print("")



print("Running 4/11");
sys.stdout = open("test_logs.txt", 'a');
system_dict = test_update_normal(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));
print("")



print("Running 5/11");
sys.stdout = open("test_logs.txt", 'a');
system_dict = test_update_eval_infer(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));
print("")



print("Running 6/11");
sys.stdout = open("test_logs.txt", 'a');
system_dict = test_expert_train(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));



print("Running 7/11");
sys.stdout = open("test_logs.txt", 'a');
system_dict = test_expert_eval_infer(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));
print("")




print("Running 8/11");
sys.stdout = open("test_logs.txt", 'a');
system_dict = test_switch_default(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));
print("")




print("Running 9/11");
sys.stdout = open("test_logs.txt", 'a');
system_dict = test_switch_expert(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));
print("")




print("Running 10/11");
sys.stdout = open("test_logs.txt", 'a');
system_dict = test_compare(system_dict)
sys.stdout = origstdout;
print("Tests Completed           - {}".format(system_dict["total_tests"]));
print("Tests Succesful           - {}".format(system_dict["successful_tests"]));
print("")


print("Running 11/11");
sys.stdout = open("test_logs.txt", 'a');
system_dict = test_analyse(system_dict)
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