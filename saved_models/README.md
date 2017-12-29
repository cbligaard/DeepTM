# Trained models for download
This folder contains two different saved models that resulted from training of the DeepTM model.
The two models were selected as the best CRF-model and the best non-CRF model (best meaning lowest validation loss).

## lr_0.001_crf_False_batch_16_model

### Saved model with settings:

* Learning rate = 0.001
* CRF = False
* Batch size = 16

Model saved during training for 1,000 epochs at epoch 420 (lowest validation loss).
The validation loss at this epoch was 1.067, and the accuracies for amino acid locations and type were 95.6 % and 97.1 %, respectively.


## lr_0.01_crf_True_batch_32_model

### Saved model with settings:

* Learning rate = 0.01
* CRF = True
* Batch size = 32

Model saved during training for 1,000 epochs at epoch 540 (lowest validation loss).
The validation loss at this epoch was 4.403, and the accuracies for amino acid locations and type were 93.5 % and 95.5 %, respectively.


The transition matrix for the CRF looked as follows:

|                    | In → out   | Out → in   | Signal peptide   | Inside    | Outside   |
|--------------------|:----------:|:----------:|:----------------:|:---------:|:---------:|
| **In → out**       |   2.596    |  -26.452   |     -22.122      | -21.082   |  4.163    |
| **Out → in**       |  -24.640   |   2.538    |     -19.531      |  4.478    |  -5.354   |
| **Signal peptide** |  -23.298   |  -30.415   |      2.206       | -10.640   |  -2.370   |
| **Inside**         |  -8.016    |  -34.591   |     -30.403      |  2.317    | -27.095   |
| **Outside**        |  -32.954   |  -9.147    |     -25.922      | -27.466   |  1.967    |
