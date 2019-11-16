from pytorch.models.imports import *
from system.imports import *
from pytorch.models.models import *
from pytorch.models.common import create_final_layer



@accepts(dict, path=[str, bool], final=bool, resume=bool, external_path=[bool, str, list], post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def load_model(system_dict, path=False, final=False, resume=False, external_path=False):
    GPUs = GPUtil.getGPUs()
    if(len(GPUs)==0):
        if(final):
            if(path):
                finetune_net = torch.load(path + "final", map_location=torch.device('cpu'));
            else:
                finetune_net = torch.load(system_dict["model_dir_relative"] + "final", map_location=torch.device('cpu'));
        if(resume):
            finetune_net = torch.load(system_dict["model_dir_relative"] + "resume_state", map_location=torch.device('cpu'));
     
        if(external_path):
            finetune_net = torch.load(external_path, map_location=torch.device('cpu'));
    else:
        if(final):
            if(path):
                finetune_net = torch.load(path + "final");
            else:
                finetune_net = torch.load(system_dict["model_dir_relative"] + "final");
        if(resume):
            finetune_net = torch.load(system_dict["model_dir_relative"] + "resume_state");
     
        if(external_path):
            finetune_net = torch.load(external_path);

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

    finetune_net, model_name = get_base_model(model_name, use_pretrained, num_classes, freeze_base_network);


    if(len(custom_network)):
        if(final_layer):
            if(model_name in set1):
                finetune_net = create_final_layer(finetune_net, custom_network, num_classes, set=1);
            elif(model_name in set2):
                finetune_net = create_final_layer(finetune_net, custom_network, num_classes, set=2);
            elif(model_name in set3):
                if(model_name == "inception_v3"):
                    msg = "Custm layer addition to inception_V3 unimplemented.\n";
                    msg += "Using basic inception_v3";
                    ConstraintWarning(msg);
                    num_ftrs = finetune_net.AuxLogits.fc.in_features;
                    finetune_net.AuxLogits.fc = nn.Linear(num_ftrs, num_classes);
                    num_ftrs = finetune_net.fc.in_features;
                    finetune_net.fc = nn.Linear(num_ftrs,num_classes);
                else:
                    finetune_net = create_final_layer(finetune_net, custom_network, num_classes, set=3);
            elif(model_name in set4):
                finetune_net = create_final_layer(finetune_net, custom_network, num_classes, set=4);
        else:
            msg = "Final layer not assigned";
            raise ConstraintError(msg);
    else:
        if(model_name in set1):
            num_ftrs = finetune_net.classifier[6].in_features;
            finetune_net.classifier[6] = nn.Linear(num_ftrs, num_classes);
        elif(model_name in set2):
            num_ftrs = finetune_net.classifier.in_features;
            finetune_net.classifier = nn.Linear(num_ftrs, num_classes);
        elif(model_name in set3):
            if(model_name == "inception_v3"):
                num_ftrs = finetune_net.AuxLogits.fc.in_features;
                finetune_net.AuxLogits.fc = nn.Linear(num_ftrs, num_classes);
                num_ftrs = finetune_net.fc.in_features;
                finetune_net.fc = nn.Linear(num_ftrs,num_classes);
            else:
                num_ftrs = finetune_net.fc.in_features;
                finetune_net.fc = nn.Linear(num_ftrs, num_classes);
        elif(model_name in set4):
            num_ftrs = finetune_net.classifier[1].in_features;
            finetune_net.classifier[1] = nn.Linear(num_ftrs, num_classes);



    system_dict["local"]["model"] = finetune_net;


    return system_dict;
