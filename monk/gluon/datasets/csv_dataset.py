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
    @accepts("self", list, list, str, post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
    def __init__(self, img_list, label_list, prefix):
        self.img_list = img_list;
        self.label_list = label_list;
        self.prefix = prefix;
    
    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)    
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



class DatasetCustomMultiLabel(Dataset):
    '''
    Class for multi label CSV dataset 

    Args:
        img_list (str): List of images 
        label_list (str): List of labels in the same order as images
        prefix (str): Path to folder containing images
        transform (torchvision transforms): List of compiled transforms
    '''
    @accepts("self", list, list, list, str, transform="self", post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
    def __init__(self, img_list, label_list, class_list, prefix):
        self.img_list = img_list;
        self.label_list = label_list;
        self.class_list = class_list;
        self.prefix = prefix;
        self.num_classes = len(self.class_list);


    @accepts("self", post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
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
        Returns transformed image and label as per index

        Args:
            None

        Returns:
            pytorch tensor: Image loaded as pytorch tensor
            int: Class ID
        '''
        image_name = self.prefix + "/" + self.img_list[index]; 
        img = image.imread(image_name);
        label_list = self.label_list[index];
        label = np.zeros((self.num_classes));
        for i in range(len(label_list)):
            index = self.class_list.index(label_list[i]);
            label[index] = 1;
        label = mx.nd.array(label);
        return img, label