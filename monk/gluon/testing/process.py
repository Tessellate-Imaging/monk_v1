from gluon.testing.imports import *
from system.imports import *

@accepts(str, bool, dict, post_trace=True)
@TraceFunction(trace_args=False, trace_rv=False)
def process_single(img_name, return_raw, system_dict):
    img = image.imread(img_name)
    img = system_dict["local"]["data_transforms"]["test"](img).expand_dims(axis=0);
    img = img.copyto(system_dict["local"]["ctx"][0]);
    outputs = system_dict["local"]["model"](img);
    if(system_dict["dataset"]["params"]["classes"]):
        prediction = system_dict["dataset"]["params"]["classes"][np.argmax(outputs[0].asnumpy())];
    else:
        prediction = np.argmax(outputs[0].asnumpy());
    score = outputs[0].asnumpy()[np.argmax(outputs[0].asnumpy())];
    if(return_raw):
        return prediction, score, outputs[0].asnumpy()
    else:
        return prediction, score, "";
    