# AiCPP

## Train set & Test set
The data generation process is shown in Figure 2. We collected data for known cell-penetrating peptides (CPPs) from several sources, including CellPPD, MLCPP, CPPsite 2.0, Lifetein(https://www.lifetein.com/), and other publications [4,17,27]. From this collection, we selected 2,798 unique peptides that were between 5 and 38 amino acids long, after removing redundant peptides and those with non-amino acid sequences. To test the performance of our model, we created a sepa-rate set of 150 CPP peptides and 150 non-CPP peptides that were not used in the other three models (MLCPP, CellPPD, CPPred). In total, we used 2,346 peptides for the train set, including 1,249 CPPs and 1,097 non-CPPs.  
  
  
To prepare the input data for our deep learning model, we used a sliding window method to slice the peptide sequences into overlapping 9-amino acid segments, as shown in Figure 3. Using a sliding window method to slice the curated peptides into overlapping 9-amino acid segments allows us to use more training data and capture local sequence patterns or meaningful sequence context features. The sliding window approach is commonly used in molecular sequence analysis to study the properties of individual residues.  
To ensure that all peptides were of uniform length, we padded shorter sequences with '-' characters to create 9-mer peptides. This step was necessary to maintain con-sistency across the dataset for the deep learning. We removed all duplicate 9-mer pep-tide sequences from the dataset.  
  
  
We generated 11,046,343 9-mer peptide fragments from 113,620 human reference proteins to be used as the negative set in the training process. By including a large number of negative datasets, we hoped to improve the model's specificity, or its ability to correctly identify non-CPPs, by reducing the bias toward predicting false positives.  
Finally, after removing duplicates in 9-mers, the AiCPP model was trained on 21,573 peptide fragments, including 7,165 positive (CPP) 9-mer peptides, 14,408 nega-tive (non-CPP) 9-mer peptides, and 11,046,343 negative 9-mer peptides derived from human reference proteins (Figure 2).  
  
## Model Algorithm
The AiCPP uses ensemble learning, which involves training multiple models and combining their predictions to obtain a more accurate and robust result. Specifically, we built five different deep-learning architectures, each including an embedding layer, a long short-term memory (LSTM) layer, and attention layers. Figure S1 and Table 1 show more detailed information about these architectures.  
To generate the input for the model, we convert the peptide sequence into a dense vector using an embedding layer. The resulting vector is used as the input for each of the five models, which use the binary cross entropy loss function and are trained for 1000 epochs using the Adam optimizer.  
To obtain a final prediction value for a given peptide sequence, we take the aver-age of the prediction values of each 9-mer obtained using a sliding window approach. Our model was implemented using Python 3.8 and TensorFlow 2.4.0.  
  
![image](https://user-images.githubusercontent.com/94620359/212254391-e7768265-10f0-410f-906c-606a057602ca.png)  
  
  
## **Performance evaluation of the CPP predictors.**
AiCPP demonstrates significantly higher performance with values of 0.927, 0.722, and 0.860 for AUC, MCC, and ACC, respectively. AiCPP also has a higher specificity for non-CPPs (0.893) compared to the three external models.  
  
![image](https://user-images.githubusercontent.com/94620359/212254581-e7c25de0-bcea-4cf3-a6a3-bf84dde806bb.png)   
  
  
