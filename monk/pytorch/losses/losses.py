from monk.pytorch.losses.imports import *
from monk.system.imports import *


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def l1(system_dict, weight=None, batch_axis=0):
    '''
    Select L1 Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "l1";
    system_dict["hyper-parameters"]["loss"]["name"] = "l1";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def l2(system_dict, weight=None, batch_axis=0):
    '''
    Select L2 Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "l2";
    system_dict["hyper-parameters"]["loss"]["name"] = "l2";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, 
    axis_to_sum_over=int, label_as_categories=bool, label_smoothing=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def softmax_crossentropy(system_dict, weight=None, batch_axis=0, axis_to_sum_over=-1, 
    label_as_categories=True, label_smoothing=False):
    '''
    Select softmax crossentropy Loss - Auto softmax before applying loss 

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N
        axis_to_sum_over (int): Set as -1
        label_as_categories (bool): Fixed as True
        label_smoothing (bool): If True, label smoothning is applied.

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "softmaxcrossentropy";
    system_dict["hyper-parameters"]["loss"]["name"] = "softmaxcrossentropy";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["loss"]["params"]["axis_to_sum_over"] = axis_to_sum_over;
    system_dict["hyper-parameters"]["loss"]["params"]["label_as_categories"] = label_as_categories;
    system_dict["hyper-parameters"]["loss"]["params"]["label_smoothing"] = label_smoothing;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, 
    axis_to_sum_over=int, label_as_categories=bool, label_smoothing=bool, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def crossentropy(system_dict, weight=None, batch_axis=0, axis_to_sum_over=-1, 
    label_as_categories=True, label_smoothing=False):
    '''
    Select crossentropy Loss - Need to manually apply softmax

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N
        axis_to_sum_over (int): Set as -1
        label_as_categories (bool): Fixed as True
        label_smoothing (bool): If True, label smoothning is applied.

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "crossentropy";
    system_dict["hyper-parameters"]["loss"]["name"] = "crossentropy";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["loss"]["params"]["axis_to_sum_over"] = axis_to_sum_over;
    system_dict["hyper-parameters"]["loss"]["params"]["label_as_categories"] = label_as_categories;
    system_dict["hyper-parameters"]["loss"]["params"]["label_smoothing"] = label_smoothing;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def sigmoid_binary_crossentropy(system_dict, weight=None, batch_axis=0):
    '''
    Select sigmoid binary crossentropy Loss - Auto sigmoid before applying loss 

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "sigmoidbinarycrossentropy";
    system_dict["hyper-parameters"]["loss"]["name"] = "sigmoidbinarycrossentropy";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;

@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def binary_crossentropy(system_dict, weight=None, batch_axis=0):
    '''
    Select binary crossentropy Loss - Need to manually apply sigmoid

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "binarycrossentropy";
    system_dict["hyper-parameters"]["loss"]["name"] = "binarycrossentropy";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


@accepts(dict, log_pre_applied=bool, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, 
    axis_to_sum_over=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def kldiv(system_dict, log_pre_applied=False, weight=None, batch_axis=0, axis_to_sum_over=-1):
    '''
    Select lkdiv Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N
        axis_to_sum_over (int): Set as -1
        log_pre_applied (bool): If set as False, then logarithmic function is auto applied over target variables

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "kldiv";
    system_dict["hyper-parameters"]["loss"]["name"] = "kldiv";
    system_dict["hyper-parameters"]["loss"]["params"]["log_pre_applied"] = log_pre_applied;
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["loss"]["params"]["axis_to_sum_over"] = axis_to_sum_over;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;

@accepts(dict, log_pre_applied=bool, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def poisson_nll(system_dict, log_pre_applied=False, weight=None, batch_axis=0):
    '''
    Select poisson_nll Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N
        log_pre_applied (bool): If set as False, then logarithmic function is auto applied over target variables

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "poissonnll";
    system_dict["hyper-parameters"]["loss"]["name"] = "poissonnll";
    system_dict["hyper-parameters"]["loss"]["params"]["log_pre_applied"] = log_pre_applied;
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, threshold_for_mean_estimator=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def huber(system_dict, weight=None, batch_axis=0, threshold_for_mean_estimator=1):
    '''
    Select huber Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N
        threshold_for_mean_estimator (int): Threshold for trimmed mean estimator.

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "huber";
    system_dict["hyper-parameters"]["loss"]["name"] = "huber";
    system_dict["hyper-parameters"]["loss"]["params"]["threshold_for_mean_estimator"] = threshold_for_mean_estimator;
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, margin=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def hinge(system_dict, weight=None, batch_axis=0, margin=1):
    '''
    Select hinge Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N
        margin (float): MArgin value.

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "hinge";
    system_dict["hyper-parameters"]["loss"]["name"] = "hinge";
    system_dict["hyper-parameters"]["loss"]["params"]["margin"] = margin;
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, margin=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def squared_hinge(system_dict, weight=None, batch_axis=0, margin=1):
    '''
    Select squared hinge Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N
        margin (float): MArgin value.

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "squaredhinge";
    system_dict["hyper-parameters"]["loss"]["name"] = "squaredhinge";
    system_dict["hyper-parameters"]["loss"]["params"]["margin"] = margin;
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, margin=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def multimargin(system_dict, weight=None, batch_axis=0, margin=1):
    '''
    Select multi margin Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N
        margin (float): MArgin value.

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "multimargin";
    system_dict["hyper-parameters"]["loss"]["name"] = "multimargin";
    system_dict["hyper-parameters"]["loss"]["params"]["margin"] = margin;
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, margin=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def squared_multimargin(system_dict, weight=None, batch_axis=0, margin=1):
    '''
    Select squared multi margin Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N
        margin (float): MArgin value.

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "squaredmultimargin";
    system_dict["hyper-parameters"]["loss"]["name"] = "squaredmultimargin";
    system_dict["hyper-parameters"]["loss"]["params"]["margin"] = margin;
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;

@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def multilabelmargin(system_dict, weight=None, batch_axis=0):
    '''
    Select multilabel margin Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "multilabelmargin";
    system_dict["hyper-parameters"]["loss"]["name"] = "multilabelmargin";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;

@accepts(dict, weight=[list, type(np.array([1, 2, 3])), float, type(None)], batch_axis=int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def multilabelsoftmargin(system_dict, weight=None, batch_axis=0):
    '''
    Select multilabel softmargin Loss

    Args:
        system_dict (dict): System dictionary storing experiment state and set variables
        weight (float): global scalar for weight loss
        batch_axis (int): Axis representing number of elements in the batch - N

    Returns:
        dict: updated system dict
    '''
    system_dict["local"]["criterion"] = "multilabelsoftmargin";
    system_dict["hyper-parameters"]["loss"]["name"] = "multilabelsoftmargin";
    system_dict["hyper-parameters"]["loss"]["params"]["weight"] = weight;
    system_dict["hyper-parameters"]["loss"]["params"]["batch_axis"] = batch_axis;
    system_dict["hyper-parameters"]["status"] = True;
    return system_dict;


