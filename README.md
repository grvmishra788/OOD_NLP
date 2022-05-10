# OOD_NLP

Repo  to maintain code for CS769 course project (UW-Madison, Spring 2022)

In this project, we proposed a new distance-based out-of-distribution detection method and an ensemble technique of combining our distance-based out-of-distribution detection with energy-based out-of-distribution detection and ODIN. We tested our proposed method on the two types of OOD data: semantic shift and background shift. Our proposed method performed much better on semantic shift OOD data. We used only post-processing and pre-processing techniques on a pre-trained model, which is easy to implement and use on the existing trained neural networks.


For running the code please follow the steps as mentioned below:
1. Create the venv module:
```
python3 -m venv /path/to/new/virtual/environment 
```
2. Activate the environment:
```
source /path/to/new/virtual/environment/bin/activate
```
3. we need to install all the dependency:
```
 pip3 install -r requirements.txt
```
4. To run the code: 
```
python3 main.py --energy_temp 5 --softmax_temp 10 --retrain False --debug False --output_folder "outputs"
```
Here, energy_temp represents temperature value for energy based OOD. \
softmax_temp represents Temperature value for temperature scaling OOD. \
retrain represents if we want to retrain the model \
debug represents if we want to print verbose information during training \
output_folder represents the output path where we want to save the outputs
