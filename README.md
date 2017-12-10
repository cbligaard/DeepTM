# DeepTM
### A deep learning setup for prediction of protein transmembrane helices

*Project, 02456 Deep Learning, Technical University of Denmark*<br>
Christina Bligaard Pedersen (s134891), Dea Steenstrup (s123870), Emma Christine Jappe (s102240) & Rasa Audange Muktupavela (s161197)

### Welcome to the repository for the DeepTM project!
This project revolved around making a deep learning model for prediction of protein transmembrane helices. The overall goal was to create a model that takes an amino acid sequence as an input and predicts the protein topology (i.e. which amino acid residues are positioned where relative to the membrane).<br>
There are five categories for each residue:<br> 
1. Membrane in-out
1. Membrane out-in
1. Signal peptide
1. Inside
1. Outside

Furthermore, the outputs a prediction of the protein type.<br>
1. Signal peptide + globular (SP+Glob)
1. Signal peptide + transmembrane (SP+TM)
1. Transmembrane (TM)
1. Globular (Glob)

#### Model
The model of choice for this problem was a bidirectional recurrent neural network (RNN) with a long short-term memory (LSTM) cell followed by a dense layer and either a conditional random field (CRF) or simply a softmax function followed by an argmax. This part of the model is used for topology prediction.
On top of this model, we average the probabilities of each category over the entire amino acid sequence length and have another two dense layers, and finally we use the sigmoid function to predict the protein type.

![Model setup](images/model.png?raw=true "Model setup: Red boxes represent the layers of the neural network, while the grey boxes represent functions used to derive the actual predictions.")

The loss function for the first part of the prediction (protein topology) is '-log_likelihood' if the CRF is used, and the 'weighted cross-entropy loss for a sequence of logits' if not. The loss function for the second part of the prediction (protein type) is the 'softmax cross entropy'. In the end, the two losses are summed and the Adam algorithm is used as the optimizer to minimize the loss. 

#### Data
The data used for this project was the TOPCONS2-data downloaded from [here](http://topcons.net/pred/download/). The data consisted of 6,856 proteins, but to ensure computational efficiency, we removed proteins longer than 2000 amino acids (n = 123), and as a result we have a dataset of 6733 proteins in total. These were divided into five partitions maintaining the same proportion of each class and the same length distribution as the full dataset. Proteins with more than 30 % homology were placed within a single partition (homology partitioning). The dataset contained 2171 SP+Glob-proteins, 718 SP+TM, 313 TM and 3531 Glob.

#### Training, validating and testing
For training, validating and testing the model, a setup was made in which 3 partitions were used for training, 1 was used for validation, and 1 was used for testing. During training, which typically ran for 1000 epochs, the model was validated every 10 epochs - and each time the validation loss decreased, the model was saved. 
Initial tests showed that using 60 units in the LSTM cells and relu as an activation function for the type-prediction, yielded good results. These settings were fixed and a grid search was performed for the following hyper-parameter value combinations (a total of 64 runs):<br>
1. With and without the CRF
1. Batch size = 16, 32, 64 or 128
1. Learning rate = 0.01, 0.005,  0.002, 0.001, 0.0005, 0.0001, 0.00005, 0.00001

After training, the model with the lowest validation loss was selected for runs with and without the CRF, and these models were tested on the test partition. 

### Files in this repository
#### Scripts for training and testing the model
1. training.py (maybe also as ipython notebook)
1. model.py
1. minibatches.py
1. confidence_plot.py
1. conf_matrix.py
1. plots.py
1. crf.py
1. marginal_probabilities.py
1. testing_model.ipynb (Jupyter notebook file for recreating the main results of this study)

#### Saved models (generated from selected training runs)
1. XXX
1. XXX
