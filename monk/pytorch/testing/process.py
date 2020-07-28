from monk.pytorch.testing.imports import *
from monk.system.imports import *

@accepts(str, bool, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def process_single(img_name, return_raw, system_dict):
    '''
    Run inference on a single image

    Args:
        img_name (str): path to image
        return_raw (bool): If True, then output dictionary contains image probability for every class in the set.
                            Else, only the most probable class score is returned back.
                            

    Returns:
        str: predicted class
        float: prediction score
    '''
    img = Image.open(img_name).convert('RGB');
    img = system_dict["local"]["data_transforms"]["test"](img);
    img = img.unsqueeze(0);
    img = Variable(img);
    img = img.to(system_dict["local"]["device"])
    outputs = system_dict["local"]["model"](img)
    l = outputs.data.cpu().numpy().argmax();
    if(system_dict["dataset"]["params"]["classes"]):
        prediction = system_dict["dataset"]["params"]["classes"][l];
    else:
        prediction = l;
    normalized = softmax(outputs.data.cpu().numpy()[0]);
    score = normalized[l];
    if(return_raw):
        return prediction, score, normalized;
    else:
        return prediction, score, "";



@accepts(str, bool, float, dict, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def process_multi(img_name, return_raw, img_thresh, system_dict):
    '''
    Run inference on a single image when label type is multi-label

    Args:
        img_name (str): path to image
        return_raw (bool): If True, then output dictionary contains image probability for every class in the set.
                            Else, only the most probable class score is returned back.
        img_thresh (float): Thresholding for multi label image classification.
                            

    Returns:
        list: list of predicted classes
        list: list of predicted scores
    '''
    img = Image.open(img_name).convert('RGB');
    img = system_dict["local"]["data_transforms"]["test"](img);
    img = img.unsqueeze(0);
    img = Variable(img);
    img = img.to(system_dict["local"]["device"])
    outputs = system_dict["local"]["model"](img)
    list_classes = [];
    raw_scores = outputs.cpu().detach().numpy()[0];
    list_scores = [];

    for i in range(len(raw_scores)):
        prob = logistic.cdf(raw_scores[i])
        if(prob > img_thresh):
            list_classes.append(system_dict["dataset"]["params"]["classes"][i])
            list_scores.append(prob)


    if(return_raw):
        return list_classes, list_scores, raw_scores;
    else:
        return list_classes, list_scores, "";