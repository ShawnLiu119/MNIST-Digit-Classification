# MNIST-Digit-Classification
MNIST-Digit-Classification


### Abstract
Handwritten digits classification has been one of fundamental but significant problems within the domain of machine learning and artificial intelligence. Many researchers have studied and contributed to patten detection algorithms to solve this issue with continuous improvement of model performance. Several compelling solutions could be built into applications to recognize handwritten digits uploaded from users’ digital devices and utilized in many fields such as check online routing bank accounts and data entry.  In this research, the MNIST dataset, a collection of hand-written scanned images, has been used to training and testing of multiple neural network models. The primary objective of this research is to experiment with neural network models with differentiated model structure and explore how the number of hidden nodes impact the model performance. Additionally, Principal Component Analysis (PCA) and Random Forest are applied for feature combination/pruning to decide the most economic size of features and find the balance between model performance and computational complexity comparing against training on full set of features.

### Introduction 
Hand-written digits recognition is one of active sub-areas of the artificial intelligence field that dedicates to classification of pre-segmented hand-written digits with learning models (Gope, Pande, Karale, Dharmale and Umekar, 2021). It has been proved to be beneficial to many application areas such as postal mail sorting and medical data entry and processing (Qiao, Wang, Li and Chen, 2018). One of challenging part of the ha-written digits identification problem is to deal with the wide variety in human writing style and extract discriminating features for categorical classification (Gope, Pande, Karale, Dharmale and Umekar, 2021). Researchers have made significant amount efforts to develop algorithms and models to tackle the problem such as deep learning based classification algorithms (Wang et al, 2016) and support vector machine classifier (Niu and Suen , 2012). The successive progress in terms of classification accuracy has been made through these efforts as more complicated and advanced algorithms and models have been developed to work on model performance optimization. According to paperswithcode.com, the highest accuracy rate of prediction is 99.91%, which was achieved by the model of Homogeneous Ensemble with Simple CNN, followed by 99.87% by Branching/Merging CNN model and 99.84% by EnsNet. 
In the research, several experiments are conducted to testify neural network with MNIST data 0as shown in Figure 1, which is one of the most fundamental collection of hand-written digit data sets to test pattern detection and image classification models. 

![image](https://user-images.githubusercontent.com/43327902/185507658-eef6cb09-d212-4a85-a697-4c1975221ebb.png)
Figure 1. Images from the MNIST training set.

The main object of conducting these experiments is to understand how variety of number of hidden nodes of neural network model impacts the feature extraction and hence model prediction. Some supplementary experiments including PCA (Principal Component Analysis) and feature importance measurement embedded in Random Forecast Algorithm, are rolled out comparing with training on full set of features in order to identify most economic size of features and understand the tradeoff between model performance and computational complexity. 

### Literature Reivew
Please refer to the final report

### Method
This section provides an overview of the proposed methodology for hand-written digits recognition experiments conducted in this study. Figure. 2 depict the overall process with key phases laid out starting from data collection to model result evaluation and interpretation.

![image](https://user-images.githubusercontent.com/43327902/185507781-6389fed3-d220-4f28-baa8-8be208a21893.png)

Figure 2. Proposed research methodology

##### Data Collection
The MNIST image datasets is assembled by the National Institute of Standards and Technology (NIST) in the 1980s. It comes with a set of 60,000 training images and 10,000 test images of processed hand-written digits from 0 to 9, which have been configured to 28 * 28 pixels in size decreased. The data is pre-loaded in Keras by Google Tensorflow and could be easily loaded as NumPy arrays. Figure 3 shows an example of digit 0 from MNIST data.  

![image](https://user-images.githubusercontent.com/43327902/185507821-3603ab07-eff2-4aac-96f4-e728e03f201a.png)

Figure 3. Example of NumPy array configured from hand-written image

##### Data Exploration and Visualization
Since the classification might be skewed by relative sample size by different digit categories, a close investigation of distribution by digit category has been conducted on both training and test datasets as demonstrated in Figure 4 and 5.  

![image](https://user-images.githubusercontent.com/43327902/185507905-84d5746b-e1d0-4541-a7ef-438dd011b393.png)

Figure 4. Training sample size by digit category

![image](https://user-images.githubusercontent.com/43327902/185507930-7e8d50a1-855d-461c-8337-74eee566418d.png)

Figure 5. Test sample size by digit category

The similarity of distribution between training and test dataset could be identified as that instances have approximately even distribution across 10 digits. Number 1 accounts for the highest number of instances (6,742 training instances, 1,135 test instances), whereas number 5 is the digit with lowest number of instances (5,421 training instances, 892 test instances). 


##### Data Preprocessing
Data normalization is a necessary step to take before the digit data is fed into classification training or feature engineering process such as PCA (Principal Component Analysis). Given all pixels are in gray-scale ranging from 0 to 255, the data is normalized to 0 to 1 simply by dividing all pixel data by 255.  

##### Feature Engineering
The baseline model has been trained on full sets of 784 pixels, in additional to which feature engineering work has been implemented to detect the most discriminating features and explore the possibility of reducing dimensions and how the consolidated features would impact the model performance. Some sample codes and visualization of PCA and Random Forest have been shown in Figure 6 and Figure 7. 

![image](https://user-images.githubusercontent.com/43327902/185508062-0575b3bd-2550-4382-a8fb-0790a64c2ed3.png)

Figure 7. Heatmap of feature importance identified by Random Forest


##### Classification
DNN (Dense Neural Network) is solely tasked to handle the classification of hand-written digits in this research as one of primary objectives of this study is to understand how hidden layer with various number of hidden nodes captures the key information differently and hence translated into model performance. Five experiments as shown in Figure 8 have been designed while experiment 1/2/3 focus on same neural 
network model structure (1 dense hidden layer) with number of hidden nodes as the only tunable hyperparameter. The alternative value of number of hidden nodes including 2, 128, 196, 392 and 532 are testified. In experiment 4, PCA (Principal Component Analysis) is brought in to transform the original features and consolidate to 154 principal components, which is fed into DNN classification model with 196 hidden nodes as identified optimal in experiment 3. In experiment 5, the pre-defined attribute of feature importance identification by random forest is utilized to prune the original pixel features to 70 based on its importance level and then fed into DNN classification model for training and test purpose. 

![image](https://user-images.githubusercontent.com/43327902/185508172-81abbf3d-337d-4ca7-b365-7fb55d9debbd.png)

Figure 8. Model Design & Structure

##### Result
DNN classification models are proposed to be trained fully in 200 epochs of training dataset, although early callbacks are set up with patience of 3 for computational efficiency. Then the trained models are applied on test dataset with its prediction precision being evaluated. Through first three experiments, five alternative number of hidden nodes have been used to tune the hidden layers of DNN model, whose prediction accuracies are evaluated as shown in Figure 9.  The model with 196 hidden nodes achieves highest test accuracy 96.77%, followed by model with 128 nodes (96.73%) and the one with 532 nodes (96.58%). 
To investigate the impact of feature engineering on model performance, DNN models with completely same structure, one hidden layer with 196 nodes derived from experiment 3, have been trained on features with reduced dimensions and compared with the model trained on full pixel features. The DNN model 
trained on 154 features transformed based on PCA surpasses the other two with 97.79% accuracy, followed by DNN model trained on full pixel data with 96.77% accuracy. 

![image](https://user-images.githubusercontent.com/43327902/185508226-f46799cd-91fb-474f-adad-336a3e5764b7.png)
![image](https://user-images.githubusercontent.com/43327902/185508236-1d258dcb-f09c-4b88-8d26-a33eac45d2ab.png)

Figure 9. Model performance comparison – DNN by different number of hidden nodes

As expected, the model that’s been trained on 70 features selected based on Random Forest, delivers 93.01% accuracy which makes sense that the trimming down of features may cause loss of information hence lower the prediction precision. 

![image](https://user-images.githubusercontent.com/43327902/185508277-f3eaa276-3a01-4edc-9d9f-4339095490c4.png)

Figure 10. Model performance comparison by different feature dimension

### Conclusion
Although it is not sufficient to asserted that 196 is the optimal number for hidden nodes setting the DNN model, we can learn from the first three experiments that the prediction accuracy does not always increase correlated with number of hidden nodes. The model with so few hidden nodes such as 2 apparently has been proved to be less capable of capturing information of all 10 digits, especially those digits with similar 
features such as 4 vs 9, or 1 vs 7, and hence achieves lower prediction accuracy. Adding hidden nodes can certainly increase the information being captured, but it will also increase the risk of overfitting and add noise into the model at the same time. For example, when the number of hidden nodes increases to 392 and 532, which accounts for ½ and 2/3 of input data dimension, the model performance has not been improved as expected but returns with slight drop of accuracy. 
Feature dimension reduction should always be considered as it reduces computation complexity without losing too much model performance. The feature importance detection by Random Forest has revealed that 70 out of all 784 features captures 95% of variance, but the benefit of reducing dimension from 784 to 70 is significant considering the number of weights and bias in neural network model. PCA has not only done a great job of consolidating features, but also improve the overall model performance by eliminating some noise.














