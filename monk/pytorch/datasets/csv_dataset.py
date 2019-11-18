from pytorch.datasets.imports import *
from system.imports import *



class DatasetCustom(Dataset):
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
        return len(self.img_list)
    
    def __getitem__(self, index):
        image_name = self.prefix + "/" + self.img_list[index];
        image = Image.open(image_name).convert('RGB');
        label = int(self.label_list[index]);       
        if self.transform is not None:
            image = self.transform(image);
        return image, label