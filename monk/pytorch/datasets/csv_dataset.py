from pytorch.datasets.imports import *
from system.imports import *



class DatasetCustom(Dataset):
    '''
    Class for single label CSV dataset 

    Args:
        img_list (str): List of images 
        label_list (str): List of labels in the same order as images
        prefix (str): Path to folder containing images
        transform (torchvision transforms): List of compiled transforms
    '''
    @accepts("self", list, list, str, transform="self", post_trace=True)
    @TraceFunction(trace_args=False, trace_rv=False)
    def __init__(self, img_list, label_list, prefix, transform=None):
        self.img_list = img_list;
        self.label_list = label_list;
        self.transform = transform;
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