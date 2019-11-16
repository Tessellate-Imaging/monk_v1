from tf_keras.models.imports import *
from system.imports import *
from tf_keras.models.models import *
from tf_keras.models.common import create_final_layer





@accepts(dict, path=[str, bool], final=bool, resume=bool, external_path=[bool, str, list], post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def load_model(system_dict, path=False, final=False, resume=False, external_path=False):
    if(final):
        if(path):
            finetune_net = keras.models.load_model(path + "final.h5");
        else:
            finetune_net = keras.models.load_model(system_dict["model_dir_relative"] + "final.h5");
    if(resume):
        finetune_net = keras.models.load_model(system_dict["model_dir_relative"] + "resume_state.h5");
 
    if(external_path):
        finetune_net = keras.models.load_model(external_path);

    return finetune_net;






@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def setup_model(system_dict):
    model_name = system_dict["model"]["params"]["model_name"];
    use_pretrained = system_dict["model"]["params"]["use_pretrained"];
    freeze_base_network = system_dict["model"]["params"]["freeze_base_network"];
    custom_network = system_dict["model"]["custom_network"];
    final_layer = system_dict["model"]["final_layer"];
    num_classes = system_dict["dataset"]["params"]["num_classes"];
    input_size = system_dict["dataset"]["params"]["input_size"];

    finetune_net, model_name = get_base_model(model_name, use_pretrained, num_classes, freeze_base_network, input_size);


    if(len(custom_network)):
        if(final_layer):
            finetune_net = create_final_layer(finetune_net, custom_network, num_classes);
        else:
            msg = "Final layer not assigned";
            raise ConstraintError(msg);
    else:
        x = finetune_net.output
        x = krl.GlobalAveragePooling2D()(x)
        x = krl.Dense(512)(x);
        x = krl.ReLU()(x)
        x = krl.Dropout(0.5)(x);
        x = krl.Dense(num_classes)(x);
        preds = krl.Softmax()(x);
        finetune_net = keras.models.Model(inputs=finetune_net.input, outputs=preds);

    system_dict["local"]["model"] = finetune_net;


    return system_dict;
