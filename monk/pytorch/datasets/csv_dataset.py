from monk.pytorch.datasets.imports import *
from monk.system.imports import *



class DatasetCustom(Dataset):
    '''
    Class for single label CSV dataset 

    Args:
        img_list (str): List of images 
        label_list (str): List of labels in the same order as images
        prefix (str): Path to folder containing images
        transform (torchvision transforms): List of compiled transforms
    '''
    @accepts("self", list, list, str, transform="self", post_trace=False)
    #@TraceFunction(trace_args=False, trace_rv=False)
    def __init__(self, img_list, label_list, prefix, transform=None):
        self.img_list = img_list;
        self.label_list = label_list;
        self.transform = transform;
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
        Returns transformed image and label as per index

        Args:
            None

        Returns:
            pytorch tensor: Image loaded as pytorch tensor
            int: Class ID
        '''
        image_name = self.prefix + "/" + self.img_list[index];
        image = Image.open(image_name).convert('RGB');
        label = int(self.label_list[index]);       
        if self.transform is not None:
            image = self.transform(image);
        return image, label




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
    def __init__(self, img_list, label_list, class_list, prefix, transform=None):
        self.img_list = img_list;
        self.label_list = label_list;
        self.class_list = class_list;
        self.transform = transform;
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
        image = Image.open(image_name).convert('RGB');  
        label_list = self.label_list[index];
        label = torch.zeros((1, self.num_classes)); 
        for i in range(len(label_list)):
            index = self.class_list.index(label_list[i]);
            label[0, index] = 1;
        if self.transform is not None:
            image = self.transform(image);
        label = torch.tensor(label, dtype= torch.float32)
        return image, label.squeeze()





