# AiCPP

## Train set & Test set
The data generation process is shown in Figure 2. We collected data for known cell-penetrating peptides (CPPs) from several sources, including CellPPD, MLCPP, CPPsite 2.0, Lifetein(https://www.lifetein.com/), and other publications[4,17,27]. From this collection, we selected 2,798 unique peptides that were between 5 and 38 amino acids long, after removing redundant peptides and those with non-amino acid sequences. To test the performance of our model, we created a sepa-rate set of 150 CPP peptides and 150 non-CPP peptides that were not used in the other three models (MLCPP, CellPPD, CPPred). In total, we used 2,346 peptides for the train set, including 1,249 CPPs and 1,097 non-CPPs.  
  
[4] Chen B, Xu W, Pan R, Chen P. Design and characterization of a new peptide vector for short interfering RNA delivery. J Nanobiotechnology. 2015;13:39. Published 2015 Jun 9.  
[17] Dobchev DA, Mager I, Tulp I, et al. Prediction of Cell-Penetrating Peptides Using Artificial Neural Networks. Curr Comput Aided Drug Des. 2010;6(2):79-89.  
[27] de Oliveira ECL, Santana K, Josino L, Lima E Lima AH, de Souza de Sales Júnior C. Predicting cell-penetrating peptides using machine learning algorithms and navigating in their chemical space. Sci Rep. 2021;11(1):7628. Published 2021 April 7.  
  
![image](https://user-images.githubusercontent.com/94620359/222610730-6b3845bb-3cbf-4430-ab11-2a5ac4a3b513.png)  
Figure 2. Dataset preparation. Peptide sequences derived from the human reference protein are used as the negative set.  
  
To prepare the input data for our deep learning model, we used a sliding window method to slice the peptide sequences into overlapping 9-amino acid segments, as shown in Figure 3. Using a sliding window method to slice the curated peptides into overlapping 9-amino acid segments allows us to use more training data and capture local sequence patterns or meaningful sequence context features. The sliding window approach is commonly used in molecular sequence analysis to study the properties of individual residues.  
To ensure that all peptides were of uniform length, we padded shorter sequences with '-' characters to create 9-mer peptides. This step was necessary to maintain con-sistency across the dataset for the deep learning. We removed all duplicate 9-mer pep-tide sequences from the dataset.  
  
![image](https://user-images.githubusercontent.com/94620359/222610802-54d3f4dc-8b8d-4187-8452-a8d0ec936a63.png)  
Figure 3. Preparation of 9-mer peptide sequences using the sliding window method for training the AiCPP model.  
  
We generated 11,046,343 9-mer peptide fragments from 113,620 human reference proteins to be used as the negative set in the training process. By including a large number of negative datasets, we hoped to improve the model's specificity, or its ability to correctly identify non-CPPs, by reducing the bias toward predicting false positives.  
Finally, after removing duplicates in 9-mers, the AiCPP model was trained on 21,573 peptide fragments, including 7,165 positive (CPP) 9-mer peptides, 14,408 nega-tive (non-CPP) 9-mer peptides, and 11,046,343 negative 9-mer peptides derived from human reference proteins (Figure 2).  
  
## Model Algorithm
The AiCPP uses ensemble learning, which involves training multiple models and combining their predictions to obtain a more accurate and robust result. Specifically, we built five different deep-learning architectures, each including an embedding layer, a long short-term memory (LSTM) layer, and attention layers. Table 1 shows more detailed information about these architectures.  
To generate the input for the model, we convert the peptide sequence into a dense vector using an embedding layer. The resulting vector is used as the input for each of the five models, which use the binary cross entropy loss function and are trained for 1000 epochs using the Adam optimizer.  
To obtain a final prediction value for a given peptide sequence, we take the aver-age of the prediction values of each 9-mer obtained using a sliding window approach. Our model was implemented using Python 3.8 and TensorFlow 2.4.0.  
  
Table 1. Five different model architectures used in AiCPP.  
![image](https://user-images.githubusercontent.com/94620359/212254391-e7768265-10f0-410f-906c-606a057602ca.png)  
  
  
## **Performance evaluation of the CPP predictors.**
Table 3 shows that the AiCPP model outperforms the other models in terms of AUC, MCC, and ACC. When evaluated using a test set, the AiCPP model outperforms MLCPP, CellPPD, and CPPred, with values of 0.927, 0.722, and 0.860 for AUC, MCC, and ACC, respectively. In addition, the AiCPP model has a higher specificity for non-CPPs (0.893) compared to the three external models listed in Table 3. Overall, these results demonstrate that the AiCPP model is a highly accurate and reliable method for predicting CPPs.  
  
Table 3. Performance evaluation of the CPP predictors. Bold numbers indicate the highest values.  
![image](https://user-images.githubusercontent.com/94620359/212254581-e7c25de0-bcea-4cf3-a6a3-bf84dde806bb.png)   
  
  
