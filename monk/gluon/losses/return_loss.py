from gluon.losses.imports import *
from system.imports import *


@accepts(dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def load_loss(system_dict):
	name = system_dict["local"]["criterion"];

	if(name == "l1"):
		system_dict["local"]["criterion"] = mx.gluon.loss.L1Loss(
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"],
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"]);

	elif(name == "l2"):
		system_dict["local"]["criterion"] = mx.gluon.loss.L2Loss(
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"],
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"]);

	elif(name == "softmaxcrossentropy"):
		system_dict["local"]["criterion"] = mx.gluon.loss.SoftmaxCrossEntropyLoss(
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"],
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"],
			axis=system_dict["hyper-parameters"]["loss"]["params"]["axis_to_sum_over"],
			sparse_label=system_dict["hyper-parameters"]["loss"]["params"]["label_as_categories"],
			from_logits=False);

	elif(name == "crossentropy"):
		system_dict["local"]["criterion"] = mx.gluon.loss.SoftmaxCrossEntropyLoss(
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"],
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"],
			axis=system_dict["hyper-parameters"]["loss"]["params"]["axis_to_sum_over"],
			sparse_label=system_dict["hyper-parameters"]["loss"]["params"]["label_as_categories"],
			from_logits=True);

	elif(name == "sigmoidbinarycrossentropy"):
		system_dict["local"]["criterion"] = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"],
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"],
			from_sigmoid=False);

	elif(name == "binarycrossentropy"):
		system_dict["local"]["criterion"] = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"],
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"],
			from_sigmoid=True);

	elif(name == "kldiv"):
		system_dict["local"]["criterion"] = mx.gluon.loss.KLDivLoss(
			from_logits=system_dict["hyper-parameters"]["loss"]["params"]["log_pre_applied"], 
			axis=system_dict["hyper-parameters"]["loss"]["params"]["axis_to_sum_over"], 
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"], 
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"]);

	elif(name == "poissonnll"):
		system_dict["local"]["criterion"] = mx.gluon.loss.PoissonNLLLoss(
			from_logits=system_dict["hyper-parameters"]["loss"]["params"]["log_pre_applied"],
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"],
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"]);

	elif(name == "huber"):
		system_dict["local"]["criterion"] = mx.gluon.loss.HuberLoss(
			rho=system_dict["hyper-parameters"]["loss"]["params"]["threshold_for_mean_estimator"],
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"],
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"]);

	elif(name == "hinge"):
		system_dict["local"]["criterion"] = mx.gluon.loss.HingeLoss(
			margin=system_dict["hyper-parameters"]["loss"]["params"]["margin"],
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"],
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"]);	

	elif(name == "squaredhinge"):
		system_dict["local"]["criterion"] = mx.gluon.loss.SquaredHingeLoss(
			margin=system_dict["hyper-parameters"]["loss"]["params"]["margin"],
			weight=system_dict["hyper-parameters"]["loss"]["params"]["weight"],
			batch_axis=system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"]);


	return system_dict;