from gluon.models.imports import *
from system.imports import *
from gluon.models.models import *
from gluon.models.common import create_final_layer


@accepts(dict, path=[str, bool], final=bool, resume=bool, external_path=[bool, str, list], post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def load_model(system_dict, path=False, final=False, resume=False, external_path=False):
    if(final):
        if(path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore");
                finetune_net = mx.gluon.SymbolBlock.imports(path + 'final-symbol.json', ['data'], path + 'final-0000.params');
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                finetune_net = mx.gluon.SymbolBlock.imports(system_dict["model_dir_relative"] + 'final-symbol.json', ['data'], system_dict["model_dir_relative"] + 'final-0000.params');
    if(resume):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore");
            finetune_net = mx.gluon.SymbolBlock.imports(system_dict["model_dir_relative"] + 'resume_state-symbol.json', ['data'], system_dict["model_dir_relative"] + 'resume_state-0000.params');
 
    if(external_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore");
            finetune_net = mx.gluon.SymbolBlock.imports(external_path[0], ['data'], external_path[1]);

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
                finetune_net = create_final_layer(finetune_net, custom_network, num_classes, set=3);
        else:
            print("Final layer not assigned");
            return 0;
    else:
        if(model_name in set1):
            with finetune_net.name_scope():
                finetune_net.output = nn.Dense(num_classes, weight_initializer=init.Xavier());
                finetune_net.output.initialize(init.Xavier(), ctx = ctx);
        elif(model_name in set2):
            net = nn.HybridSequential();
            with net.name_scope():
                net.add(nn.Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), weight_initializer=init.Xavier()));
                net.add(nn.Flatten());
            with finetune_net.name_scope():
                finetune_net.output = net;
                finetune_net.output.initialize(init.Xavier(), ctx = ctx)
        elif(model_name in set3):
            with finetune_net.name_scope():
                finetune_net.fc = nn.Dense(num_classes, weight_initializer=init.Xavier());
                finetune_net.fc.initialize(init.Xavier(), ctx = ctx)


    if(not use_pretrained):
        finetune_net.initialize(init.Xavier(), ctx = ctx)


    system_dict["local"]["model"] = finetune_net;

    return system_dict;

