from tf_keras.testing.imports import *
from system.imports import *

@accepts(str, bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def process_single(img_name, return_raw, system_dict):
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








