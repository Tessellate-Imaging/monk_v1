import os
import sys

import psutil

from monk.pytorch_prototype import prototype
from monk.compare_prototype import compare
from monk.pip_functionality_tests.pytorch.common import print_start
from monk.pip_functionality_tests.pytorch.common import print_status



def test_expert_eval_infer(system_dict):
    forward = True;
    if(not os.path.isdir("datasets")):
        os.system("! wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rG-U1mS8hDU7_wM56a1kc-li_zHLtbq2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rG-U1mS8hDU7_wM56a1kc-li_zHLtbq2\" -O datasets.zip && rm -rf /tmp/cookies.txt")
        os.system("! unzip -qq datasets.zip")

    test = "expert_eval_infer_object_creation";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            ptf = prototype(verbose=0);
            system_dict["successful_tests"] += 1;
            print_status("Pass");
        except Exception as e:
            system_dict["failed_tests_exceptions"].append(e);
            system_dict["failed_tests_lists"].append(test);
            forward = False;
            print_status("Fail");
    else:
        system_dict["skipped_tests_lists"].append(test);
        print_status("Skipped");


    test = "expert_eval_infer_Prototype()";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            ptf.Prototype("sample-project-1", "sample-experiment-4", eval_infer=True);
            system_dict["successful_tests"] += 1;
            print_status("Pass");
        except Exception as e:
            system_dict["failed_tests_exceptions"].append(e);
            system_dict["failed_tests_lists"].append(test);
            forward = False;
            print_status("Fail");
    else:
        system_dict["skipped_tests_lists"].append(test);
        print_status("Skipped");


    test = "expert_eval_infer_Dataset_Params()";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            ptf.Dataset_Params(dataset_path="datasets/dataset_csv_id/test", 
                path_to_csv="datasets/dataset_csv_id/test.csv", delimiter=",");
            system_dict["successful_tests"] += 1;
            print_status("Pass");
        except Exception as e:
            system_dict["failed_tests_exceptions"].append(e);
            system_dict["failed_tests_lists"].append(test);
            forward = False;
            print_status("Fail");
    else:
        system_dict["skipped_tests_lists"].append(test);
        print_status("Skipped");


    test = "expert_eval_infer_reset_transforms()";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            ptf.reset_transforms(test=True);
            system_dict["successful_tests"] += 1;
            print_status("Pass");
        except Exception as e:
            system_dict["failed_tests_exceptions"].append(e);
            system_dict["failed_tests_lists"].append(test);
            forward = False;
            print_status("Fail");
    else:
        system_dict["skipped_tests_lists"].append(test);
        print_status("Skipped");


    test = "expert_eval_infer_Dataset()";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            ptf.Dataset();
            system_dict["successful_tests"] += 1;
            print_status("Pass");
        except Exception as e:
            system_dict["failed_tests_exceptions"].append(e);
            system_dict["failed_tests_lists"].append(test);
            forward = False;
            print_status("Fail");
    else:
        system_dict["skipped_tests_lists"].append(test);
        print_status("Skipped");


    test = "expert_eval_infer_Evaluate()";
    system_dict["total_tests"] += 1;
    print_start(test, system_dict["total_tests"])
    if(forward):
        try:
            accuracy, class_based_accuracy = ptf.Evaluate();
            system_dict["successful_tests"] += 1;
            print_status("Pass");
        except Exception as e:
            system_dict["failed_tests_exceptions"].append(e);
            system_dict["failed_tests_lists"].append(test);
            forward = False;
            print_status("Fail");
    else:
        system_dict["skipped_tests_lists"].append(test);
        print_status("Skipped");


    return system_dict
