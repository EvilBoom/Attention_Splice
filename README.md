# Attention Splice
******
Source code for paper: AttentionSplice: An interpretable multi-head self-attention based hybrid deep learning model in splice site prediction
Detecting splice sites is important for current DNA annotation system 
and challenges the conventional methods. We propose the AttentionSplice
model, which combines multi-head self-attention and hybrid deep
learning construction to identify the splice sites. 
We extract important positions and key motifs which could be 
essential for splice site detection.


## Requirements  
******
We conduct our experiments on the following environments:
```
python == 3.7  
torch == 1.8.0     
cuda == cu111     
transformer == 4.7.0     
gpu: GeForce RTX 3090 
```
## Datasets
******
we use Human Nuclear DNA sequence data and annotations 
of the corresponding sequences were acquired from HS3D.you can find this dataset in <http://www.sci.unisannio.it/docenti/rampone/>  
we also use the caenorhabditis elegans dataset (CED), you can find this dataset in  <https://public.bmi.inf.ethz.ch/user/behr/splicing/C_elegans/>

# How to run
******
1. You can put the data files in the datasets folder
2. run p_attention in /attention/pytorch_attention
```shell
python p_attention.py
```
3. log_file will save under pytorch_attention folder