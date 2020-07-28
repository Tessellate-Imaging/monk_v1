from monk.gluon.datasets.imports import *
from monk.system.imports import *

@accepts(list, int, post_trace=False)
#@TraceFunction(trace_args=False, trace_rv=False)
def balance_class_weights(label_list, nclasses):                        
    count = [0] * nclasses
    pbar=tqdm(total=len(label_list));
    for idx, val in enumerate(label_list):       
        pbar.update();
        count[val] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(label_list)   
    pbar=tqdm(total=len(label_list));
    for idx, val in enumerate(label_list):  
        pbar.update();
        weight[idx] = weight_per_class[val]                                 
    return weight, weight_per_class, count;