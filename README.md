[![DOI:10.24433/CO.5429205.v1](http://img.shields.io/badge/DOI-10.24433/CO.5429205.v1-B31B1B.svg)](https://doi.org/10.24433/CO.5429205.v1)
# Learning finite difference methods for reaction-diffusion type equations with FCNN

[Yongho Kim and Yongho Choi (2022) *Learning Finite Difference Methods for Reaction-Diffusion Type Equations with FCNN*, Computers and Mathematics with Applications, Computers and Mathematics with Applications](https://doi.org/10.1016/j.camwa.2022.08.006)

In this paper, we propose Five-point stencil CNN (FCNN) containing a five-point stencil kernel and a trainable approximation function. We consider reaction-diffusion type equations including heat, Fisher’s, Allen-Cahn equations, and reaction-diffusion equations with trigonometric functions terms. We show that FCNNs are trained well using few data (used only two consecutive data) and then can predict reaction-diffusion evolutions with diverse initial conditions. 




## Five-point stencil Convolutional Neural Networks (FCNNs)
<p align="center">
<img width="400" alt="modelarchitecture" src="https://user-images.githubusercontent.com/52735725/147872828-452f41d8-5a86-4803-9b34-ba88b57fa99f.png">
  
## 1. Train
### 1.1 Hyperparameters  
```python
"""
--eq: 'he', 'fe', 'ac', 'sine', 'tanh' (str)
--c: diffusion coefficient (float)
--r: reaction coefficient (float)
--numepochs: number of epochs (int)
--sig: standard deviation for noise generation (float)  
--poly_order: order of polynomial approximation (int)
--lr: learning rate (float)
--pretrained: pretrained model path (str)  
"""
```
### 1.2 Execution  
```
python train.py --eq sine --poly_order 9 --r 40 --c 0.1
```
## 2. Evaluation 
Relative L2 error with the 95% confidence interval over 100 different random initial values
### 2.1 Hyperparameters  
```python
"""
--eq: 'he', 'fe', 'ac', 'sine', 'tanh' (str)
--c: diffusion coefficient (float)
--r: reaction coefficient (float)
--max_iter: max iteration (int)
--poly_order: order of polynomial approximation (int)
--pretrained: pretrained model path (str)  
"""
```  
### 2.2 Execution  
Once your model is trained you can evaluate the pretrained model. (Check your hyperparameters)
```
python evaluation.py --eq fe --r 20 --pretrained './models/fe/fe_3_0.pth' --poly_order 3
```  
### 2.3 Result
<p align="center">
<img width="500" alt="modelarchitecture" src="https://user-images.githubusercontent.com/52735725/147873261-ae19930c-ec2f-4995-8e92-b2b5939dc47f.png">  
  
## 3. Test with different initial shapes
### 3.1 Hyperparameters  
```python
"""
--eq: 'he', 'fe', 'ac', 'sine', 'tanh' (str)
--init: 'circle', 'star', 'threecircles', 'maze', 'torus' (str)
--c: diffusion coefficient (float)
--r: reaction coefficient (float)
--max_iter: max iteration (int) 
--poly_order: order of polynomial approximation (int)
--pretrained: pretrained model path (str)  
"""
```  
### 3.2 Execution  
Once your model is trained you can evaluate the pretrained model. (Check your hyperparameters)
```
python test.py --eq ac --init star --r 6944 --pretrained './models/ac/ac_3_0.pth' --max_iter 2500
```    
### 3.3 Result
<p align="center">
<img width="500" alt="modelarchitecture" src="https://user-images.githubusercontent.com/52735725/147873278-b174c95d-8708-4cb5-a1e5-da96d4cca0e9.png">    

<p align="center">
<img width="500" alt="modelarchitecture" src="https://user-images.githubusercontent.com/52735725/147873193-4a4c5571-66e9-4d96-bab5-5f1d01f2942d.png">    
  
  
  
## * License
Code [MIT license](https://opensource.org/licenses/MIT)
