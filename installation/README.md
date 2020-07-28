## PIP Installation Instructions (Recommended)

<br />

### Complete Set
  - **CPU** (Non GPU)&nbsp;: `pip install -U monk-cpu`
  - **CUDA 9.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-cuda90`
  - **CUDA 9.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-cuda92`
  - **CUDA 10.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-cuda100`
  - **CUDA 10.1** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-cuda101`
  - **CUDA 10.2** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-cuda102`
  - **Google Colab** &nbsp;&nbsp;: `pip install -U monk-colab`
  - **Kaggle** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-kaggle`

<br />
<br />

### Only Mxnet-Gluon Backend
  - **CPU** (Non GPU)&nbsp;: `pip install -U monk-gluon-cpu`
  - **CUDA 9.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-gluon-cuda90`
  - **CUDA 9.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-gluon-cuda92`
  - **CUDA 10.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-gluon-cuda100`
  - **CUDA 10.1** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-gluon-cuda101`
  - **CUDA 10.2** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-gluon-cuda102`
  - **Google Colab** &nbsp;&nbsp;: `pip install -U monk-gluon-colab`
  - **Kaggle** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-gluon-kaggle`


<br />
<br />

### Only Pytorch Backend
  - **CPU** (Non GPU)&nbsp;: `pip install -U monk-pytorch-cpu`
  - **CUDA 9.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-pytorch-cuda90`
  - **CUDA 9.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-pytorch-cuda92`
  - **CUDA 10.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-pytorch-cuda100`
  - **CUDA 10.1** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-pytorch-cuda101`
  - **CUDA 10.2** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-pytorch-cuda102`
  - **Google Colab** &nbsp;&nbsp;: `pip install -U monk-pytorch-colab`
  - **Kaggle** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-pytorch-kaggle`
  

<br />
<br />

### Only Keras Backend
  - **CPU** (Non GPU)&nbsp;: `pip install -U monk-keras-cpu`
  - **CUDA 9.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-keras-cuda90`
  - **CUDA 9.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-keras-cuda92`
  - **CUDA 10.0** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-keras-cuda100`
  - **CUDA 10.1** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-keras-cuda101`
  - **CUDA 10.2** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-keras-cuda102`
  - **Google Colab** &nbsp;&nbsp;: `pip install -U monk-keras-colab`
  - **Kaggle** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: `pip install -U monk-keras-kaggle`


<br />
<br />

### Run the following python snippets to check installation (Optional)

```python
# Unit tests for gluon backend
from monk.gluon_tests import run_unit_tests
run_unit_tests()
```

```python
# Unit tests for pytorch backend
from monk.pytorch_tests import run_unit_tests
run_unit_tests()
```

```python
# Unit tests for keras backend
from monk.keras_tests import run_unit_tests
run_unit_tests()
```

```python
# System functionality tests for gluon backend
from monk.gluon_tests import run_functionality_tests
run_functionality_tests()
```

```python
# System functionality tests for pytorch backend
from monk.pytorch_tests import run_functionality_tests
run_functionality_tests()
```

```python
# System functionality tests for keras backend
from monk.keras_tests import run_functionality_tests
run_functionality_tests()
```




<br />
<br />
<br />

# Manually installing the library (Not Recommended)


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
  


## Miscellaneous

    - Kaggle
    
          pip install -r Misc/requirements_kaggle.txt
          
    
    - Colab
    
          pip install -r Misc/requirements_colab.txt
          
