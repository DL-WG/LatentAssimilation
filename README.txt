Each script takes parameters in input: read the file Command.txt to see the commands to run the script.
Each folder contains a Command.txt file in which there are the ordered instructions.

DataSet
The data created with Fluidity are serialized and stored in the file 'data' at the path DataSet/Structured/.
At the same path are stored also the normalized data in the file 'scaled' and the train and test dataset.

Autoencoder
All the configurations of the Autoencoder tested are stored at the path AutoEncoder/Structured/LS7/.
For example, we first analyzed how many Convolutional Layers the autoencoder should have. In the file AutoEncoder/Structured/LS7/ExpLayers.py the are three different autoencoder structures and it's specified the path where the results must be stored also (in this case, the results are stored in the folder AutoEncoder/Structured/LS7/ResultLayers).
We used the script AutoEncoder/Experiments.py to test the different configurations.
Finally, for the Grid Search we used the script GridSearchAE.py
The results are analyzed in the notebook AutoEncoder/AnalysisLS7.ipynb.
The trained Autoencoder is stored at the path AutoEncoder/Structured/LS7/model-64-relu-32-400.h5.

LSTM
The train and validation set for the LSTM are created with the script train_val.py and they are stored at the path LSTM/Structured/.
We tested different configurations of LSTM with the script LSTM/LSTM_Layers.py and we perform the Grid Search with the script LSTM/LSTM_GS.py.
All results are stored at the path LSTM/Structured/ and they are analyezed in the notebook AutoEncoder/AnalysisLS7.ipynb.
The final LSTM is trained through the script fit_lstm.py and it is stored at the path LSTM/Structured/model-3-elu-30-400-16.h5.

DataAssimilation
The data of the observations are stored at the path DataAssimilation/SensorsData/Observation.xlsx.
The script LatentAssimilation.py performs the assimilation in the Latent Space and it prints the table of the results.

Increasing the latent space
To increase the latent space, you should train the autoencoder, the LSTM and, finally, you can use the Latent Assimilation model. 
Assuming the latent space size equal to 1000, the commands are:
1. AutoEncoder
	python3 AutoEncoder.py 1000 ./Structured/LS7/GridSearch.py ../DataSet/Structured/train ./Structured/LS1000/ 64 relu 32 400 
2. LSTM 
	python3 train_val.py ../DataSet/Structured/train ../AutoEncoder/Structured/LS1000/model-64-relu-32-400-1000.h5 1000 ./Structured/LS1000/
	python3 fit_lstm.py Structured/LS1000/ 1000 3 elu 30 400 16 Structured/LS1000/
3. Latent Assimilation
	python3 LatentAssimilation.py 1000 ../AutoEncoder/Structured/LS1000/model-64-relu-32-400-1000.h5 ../LSTM/Structured/LS1000/model-3-elu-30-400-16.h5 ../DataSet/Structured/test Structured/scaled
