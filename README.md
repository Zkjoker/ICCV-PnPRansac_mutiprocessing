# PnPRansac-mutiprocessing
Reproduction and revision of solvePnPRansac algorithm

## Purpose<br>
1. This program is used in a project in my lab.<br><br>
2. In this project, we change the PnPRansac algorithm and make it finishing iteration only when the current Rotation Matrix or Translation Vector has little change with last Rotation Matrix or Translation Vector no more than a threshold which can be set by user freely rather than finishing iteration when the algorithm reach the iteration times.<br><br>

## Notice<br>
1. In our project this program is put in a neural network latter so the type of many variables is torch.Tensor().In the program the input can be numpy.array() or t.Tensor(). But if you only use numpy.array(),you'd better to delete the related t.Tensor() variable for coding's efficiency.<br><br>
2. For fast running speed,We support two version of this algorithm: muti-ProReRansac.py and ProReRansac.py.The former is running for mutiprocessing(**Recommend to use this**).Of course you can change my code and make it available for cuda.<br><br>
3. In the code, notes are written by Chinese and output the state of system continuously.If you have any trouble for them please feel free to contact me.I will answer you soon.<br><br>
4. Pytorch 4.0 or higher version is ok,and it's usable for any numpy version.<br><br>

