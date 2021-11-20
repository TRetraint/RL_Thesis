# RL_Thesis
Codes and Report of the thesis made in collaboration with Airbus. If you want to understand the project I recommend you to read the report before.

This document is the READ_ME of this repository. It explains how to run the different Python scripts present.

It is first necessary to install a version of the Python interpreter > 3.6. Here is the link to download Python.

Then, it is necessary to install all the libraries that are used in the different scripts. For this, the requirements.txt file is present in the folder. To install the libraries, you have to type the command in the Python interpreter:

pip install -r requirements.txt

This command may take several minutes to complete.
In the root of the folder, you can find 2 folders and the thesis report.
The first folder is the Landing Simulator. It is broken down into 2 .py files. The first one, forces.py, computes the forces taken into account for the landing simulation.

The second file calculates the successive states of the aircraft over time and then plots the trajectories of the aircraft.

The second file is dedicated to the Reinforcement Learning scripts. It contains the file test_OpenAI_gym.py which will launch the visualisation of the Cart-Pole environment developed by the company. This file is used as a test for the correct installation of the libraries. If it runs normally, your Python interpreter is well configured.

We also find the files LandingEnv.py and LandingModel.py which respectively correspond to the environment implemented on TensorForce and the simulator.

Then, we can find the file runner.py which allows to launch the training of the models. The main.py file contains the optimized agent. Finally, the params_tuning.py file contains the code allowing to launch the optimisation of the parameters. However, the execution of the script takes a long time before it is finished.
The different files are empty but will fill up after the execution of the different scripts with the model backups and the produced data.