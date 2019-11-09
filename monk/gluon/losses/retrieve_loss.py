from gluon.losses.imports import *
from system.imports import *


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def retrieve_loss(system_dict):
	system_dict["local"]["criterion"] = system_dict["hyper-parameters"]["loss"]["name"];
	name = system_dict["local"]["criterion"];
	if(name == "softmaxcrossentropy"):
		system_dict["local"]["criterion"] = mx.gluon.loss.SoftmaxCrossEntropyLoss(weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"]);
	elif(name == "sigmoidbinarycrossentropy"):
		system_dict["local"]["criterion"] = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True, weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"]);
	elif(name == "binarycrossentropy"):
		system_dict["local"]["criterion"] = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"]);
	elif(name == "poissonnll"):
		system_dict["local"]["criterion"] = mx.gluon.loss.PoissonNLLLoss(weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"]);

	return system_dict;