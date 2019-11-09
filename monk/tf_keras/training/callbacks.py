from tf_keras.training.imports import *
from system.imports import *


class TimeHistory(krc.Callback):
    def __init__(self, log_dir=None):
        super().__init__()
        if(log_dir):
            self.log_file = log_dir + "times.txt";
            self.f = open(self.log_file, 'a');
        else:
            self.log_file=None
        

    def on_train_begin(self, logs={}):
        self.times = [];        

    def on_train_end(self, logs={}):
        if(self.log_file):
            self.f.close();        

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        if(self.log_file):
            self.f.write(str(time.time() - self.epoch_time_start) + "\n");



class MemoryHistory(krc.Callback):
    def __init__(self):
        super().__init__()
        self.max_gpu_usage=0;
        
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, batch, logs={}):
        return
 
    def on_epoch_end(self, batch, logs={}):
        import GPUtil
        GPUs = GPUtil.getGPUs()
        if(len(GPUs) > 0):
            gpuMemoryUsed = GPUs[0].memoryUsed
            if(self.max_gpu_usage < int(gpuMemoryUsed)):
                self.max_gpu_usage = int(gpuMemoryUsed);
        return