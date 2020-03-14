from gluon.datasets.imports import *
from system.imports import *



class DatasetCustom(Dataset):
    '''
    Class for single label CSV dataset 

    Args:
        img_list (str): List of images 
        label_list (str): List of labels in the same order as images
        prefix (str): Path to folder containing images
    '''
    @accepts("self", list, list, str, post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def __init__(self, img_list, label_list, prefix):
        self.img_list = img_list;
        self.label_list = label_list;
        self.prefix = prefix;
    
    @accepts("self", post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)    
    def __len__(self):
        '''
        Returns length of images in dataset

        Args:
            None

        Returns:
            int: Length of images in dataset
        '''
        return len(self.img_list)
    
    def __getitem__(self, index):
        '''
        Returns image and label as per index

        Args:
            None

        Returns:
            mxnet image: Image loaded as mx-ndarray
            int: Class ID
        '''
        image_name = self.prefix + "/" + self.img_list[index];
        img = image.imread(image_name);
        label = int(self.label_list[index]);       
        return img, label

