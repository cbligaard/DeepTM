# DeepTM
### A deep learning setup for prediction of protein transmembrane helices

*Project, 02456 Deep Learning*<br>
Christina Bligaard Pedersen, Dea Steenstrup, Emma Christine Jappe & Rasa Audange Muktupavela

### Welcome to the repository for the DeepTM project!
This project revolved around making a deep learning model for prediction of protein transmembrane helices. The overall goal was to create a model that takes an amino acid sequence as an input and predicts the protein topology (i.e. which amino acid residues are positioned where relative to the membrane). Furthermore it outputs a prediction of the protein type: Transmembrane or not.<br>
There are five categories for each residue:<br> 
1. Membrane in-out
1. Membrane out-in
1. Signal peptide
1. Inside
1. Outside

#### Model
The model of choice for this problem was a bidirectional recurrent neural network (RNN) with a long short-term memory (LSTM) cell followed by a dense layer and either a conditional random field (CRF) or simply a softmax function followed by an argmax. This part of the model is used for topology prediction.
On top of this model, we average the probabilities of each category over the entire amino acid sequence length and have another two dense layers, and finally we use the sigmoid function to predict the protein type.

Inline-style: 
![Model setup](https://github.com/cbligaard/DeepTM/master/model.png "Model setup. Red boxes represent the layers of the neural network, while the grey boxes represent functions used to derive the actual predictions.")


The loss function for the first part of the prediction (protein topology) is '-log_likelihood' if the CRF is used and the 'weighted cross-entropy loss for a sequence of logits' if not. The loss function for the second part of the prediction (protein type) is the 'sigmoid cross entropy'. In the end, the two losses are summed and the Adam algorithm is used as the optimizer to minimize the loss. 



##### Model details 
###### Recurrent neural networks (RNNs)

###### Long short-term memory (LSTM)

###### Conditional random fields (CRFs)


#### Training
For training the model we implemented a four-fold cross-validation set-up for which we use a total of five partitions. 

### Scripts
1. training.py
1. model.py
1. createMiniBatches.py
1. crf.py
1. marginal_probabilities.py
