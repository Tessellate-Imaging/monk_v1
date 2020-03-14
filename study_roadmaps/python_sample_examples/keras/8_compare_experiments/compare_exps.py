import os
import sys
sys.path.append("../../../monk/");
import psutil


from compare_prototype import compare



ctf = compare(verbose=1);
ctf.Comparison("Sample-Comparison-1");
ctf.Add_Experiment("sample-project-1", "sample-experiment-1");
ctf.Add_Experiment("sample-project-1", "sample-experiment-2");
ctf.Add_Experiment("sample-project-1", "sample-experiment-3");
ctf.Add_Experiment("sample-project-1", "sample-experiment-4");
ctf.Add_Experiment("sample-project-1", "sample-experiment-5");
ctf.Add_Experiment("sample-project-1", "sample-experiment-6");

ctf.Generate_Statistics();