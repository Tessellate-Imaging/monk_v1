# Installation Instructions

Note: 
  
        pip commands mentioned below are to be run from parent directory monk_v1/installation. 
  
        If you are running it from a  different directory, accordingly adjust the relative path to requirements files



## Linux 
  
  - CPU: 
        
        pip install -r Linux/requirements_cpu.txt
  
  - CPU with tensorflow 2.0 support:
        
        pip install -r Linux/requirements_tensorflow2_cpu.txt
  
  - GPU with Cuda-9: 
        
        pip install -r Linux/requirements_cu9.txt
  
  - GPU with Cuda-10: 
        
        pip install -r Linux/requirements_cu10.txt
  
  - GPU with tensorflow 2.0 support:
        
        pip install -r Linux/requirements_tensorflow2_gpu.txt
        
        
## MacOS

   - CPU:
   
         pip install -r Mac/requirements_cpu_macos.txt
        
        
   - GPU:
         (Not available)
          
          
          
## Windows

   - CPU:
    
         pip install -r Windows/requirements_cpu_windows.txt
         pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
        
        
   - GPU:
    
          (In development)
  

