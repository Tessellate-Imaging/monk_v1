from monk.tf_keras_1.testing.imports import *
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
    input_size = system_dict["dataset"]["params"]["input_size"];
    normalize = system_dict["local"]["normalize"];
    mean_subtract = system_dict["local"]["mean_subtract"];
    mean = system_dict["local"]["transforms_test"]["mean"];
    std = system_dict["local"]["transforms_test"]["std"];

    img = cv2.imread(img_name, 1);
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    img = img.astype(np.float32);

    if(normalize):
        img = img-mean;
        img = img/std;
    elif(mean_subtract):
        img = img-mean;
    img = np.expand_dims(img, axis=0)



    output = system_dict["local"]["model"].predict(img);

    if(system_dict["dataset"]["params"]["classes"]):
        class_names = list(system_dict["dataset"]["params"]["classes"].keys());
        prediction = class_names[np.argmax(output)];
    else:
        prediction = np.argmax(output);

    score = output[0][np.argmax(output)];

    if(return_raw):
        return prediction, score, output[0];
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

    input_size = system_dict["dataset"]["params"]["input_size"];
    normalize = system_dict["local"]["normalize"];
    mean_subtract = system_dict["local"]["mean_subtract"];
    mean = system_dict["local"]["transforms_test"]["mean"];
    std = system_dict["local"]["transforms_test"]["std"];

    img = cv2.imread(img_name, 1);
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    img = img.astype(np.float32);

    if(normalize):
        img = img-mean;
        img = img/std;
    elif(mean_subtract):
        img = img-mean;
    img = np.expand_dims(img, axis=0)



    outputs = system_dict["local"]["model"].predict(img);

    list_classes = [];
    raw_scores = outputs[0];
    list_scores = [];

    for i in range(len(raw_scores)):
        prob = logistic.cdf(raw_scores[i])
        if(prob > img_thresh):
            list_classes.append(list(system_dict["dataset"]["params"]["classes"].keys())[i])
            list_scores.append(prob)


    if(return_raw):
        return list_classes, list_scores, raw_scores;
    else:
        return list_classes, list_scores, "";


