# AiCPP

These are the seven AiCPP models that we developed.

## Dataset

### Train set
For training our CPP model, 3894 sequences with 2729 CPPs and 1165 non-CPPs were collected from the MLCPP (http://www.thegleelab.org/MLCPP/), CPPsite 2.0 (http://crdd.osdd.net/raghava/cppsite/), and lifetein (https://www.lifetein.com/Cell_Permeable_Peptides.html) databases. In addition, 113,620 human reference proteins  was extracted for use as decoys.

### External validation set
To validate our model, 2048 sequences with 806 CPPs and 1242 non-CPPs were collected from the CellPPD (https://webs.iiitd.edu.in/raghava/cellppd/index.html) database and three articles (Dobchev et al., 2010; de Oliveira et al., 2021; Chen et al., 2015). Duplicate sequences or sequences in the training set were removed and, finally, 453 sequences with 109 CPPs and 344 non-CPPs were used for model validation.


## Model Algorithm
For six of the seven models, except for model 7, the CPP predicted value of a given peptide sequence was calculated as the average of the sum of the predicted values of each 9mer obtained by sliding one 9mer from the beginning of the sequence to predict the CPP probability. 

In addition, the CPP predicted value of one amino acid was calculat-ed as the average of the predicted values of 9mers including the amino acid. The six models included an embedding, long short-term memory, and attention layer (Table 2). The last model 7 calculated the classifica-tion prediction value using the transformer encoder for the 32mer se-quence.

In each of the seven models, the peptide sequence was converted to a 3–10-dimensional dense vector using the embedding layer, which was then used as the model input. Each model used binary cross entropy as a loss function and an adaptive moment estimation optimizer.
  
  
## **Comparison of the seven model architectures.**
![image](https://user-images.githubusercontent.com/94620359/181461485-57170fa3-40b3-43ac-acb7-88692fc6f6be.png)
  
  
## **Performance evaluation of the CPP predictors.**
![image](https://user-images.githubusercontent.com/94620359/181497624-89e1cb9f-d96d-44f0-86b9-f0b320a47724.png)

