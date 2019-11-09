from tf_keras.transforms.imports import *
from tf_keras.transforms.common import set_transforms
from system.imports import *


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_trainval_transforms(system_dict):
    set_phases = ["train", "val"];
    system_dict = set_transforms(system_dict, set_phases);
    return system_dict;


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_test_transforms(system_dict):
    set_phases = ["test"];
    system_dict= set_transforms(system_dict, set_phases);
    return system_dict;