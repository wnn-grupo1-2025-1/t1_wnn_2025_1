# wisardpkg - A library for WiSARD-based models  

Aluı´zio S. Lima Filho $^ { 1 }$ , Gabriel P. Guarisa $\cdot ^ { 1 }$ , Leopoldo A. D. Lusquino Filho $^ { 1 }$ , Luiz F. R. Oliveira $^ { 1 } , { }$ , Felipe M. G. Fran¸ca $\cdot ^ { 1 }$ , Priscila M. V. Lima $^ { 1 } , ^ { }$ 2 $^ { 1 }$ PESC/COPPE/UFRJ, 2 NCE/UFRJ  

May 5, 2020  

# Abstract  

In order to facilitate the production of codes using WiSARD-based models, LabZero developed an ML library C++/Python called wisardpkg. This library is an MIT-licensed open-source package hosted on GitHub under the license.  

# 1 Introduction  

Weightless artificial neural networks (WANN) are neural models that do not use weighted synapses to store the information it learns from presented patterns. Alternatively, it possesses RAM (random-access-memory)-based neurons in which information storage takes place. In a WANN, learning of a pattern corresponds to writing in memory, whereas classification essentially corresponds to the reading of certain memory positions. The advantages of these models lie essentially in their speed, simplicity, low computational and power costs and, above all, in their ability to perform online learning.  

The WiSARD (Wilkes, Stonham and Aleksander Recognition Device) [1] is a pioneering WANN model that was originally designed to solve simple classification tasks. WiSARD has received extensions to perform semi-supervised and unsupervised learning, regression tasks, and improvements in training and classification policies.  

Recent applications of WiSARD with remarkable results corroborating the choice of this model: data stream clustering [4, 5, 6]; time-series classification [7]; audio processing [8]; online tracking of objects [9]; GPS trajectory classification [10]; part-of-speech tagging [11, 12]; text categorization [14]; hardware assisted security [15]; emotional analysis [16, 17, 18, 19], and; prediction tasks [3, 20]. Also, recent studies on theoretical aspects of the model can be found in [13].  

To facilitate the application of WiSARD and its extensions, a library in C/C++ was created, with a wrapper for Python, called wisardpkg. Section 02 of this work describes the models present in wisardpkg, while Section 03 gives details of their implementation, use, configurations and installation.  

![](images/413d07520fbb3b458df1245d752ecd99ad0a70049dbb049afa4d6d2d9295cdc6.jpg)  
Figure 1: Training in WiSARD[21].  

# 2 WiSARD-models in wisardpkg  

# 2.1 WiSARD  

WiSARD[1] is a $n$ -tuple classifier composed by class discriminators; each discriminator is a set of $N$ RAM nodes having $n$ address lines each. All discriminators share a structure called input retina, from which a pseudo-random mapping of its $N * n$ bits composes the input address lines of all of its RAM nodes.  

WiSARD is initialized with all its memory locations with value “0”. During training, when a binary pattern is presented to the network, it will access the corresponding memory positions in the appropriate discriminator, changing those with null content to “1”. This process is showed in Fig. 1. In the classification phase, the binary pattern accesses all discriminators in the corresponding positions and each of them will return a score formed by the number of non-null positions accessed. The discriminator with the highest score will determine the class of the input.  

An extension to the model was made to deal with a learning saturation problem. It consists of replacing the content of the memory positions from a singe bit to an access counter, which is increased during the training phase. In the classification phase, the discriminators’ score becomes the sum of the non-null memory positions accessed, whose counter has a value higher than a threshold called bleaching[2], which is initialized with a value of “0” and is increased whenever there is a tie. In this case, the classification process is repeated until there is a tie and if the value of bleaching becomes greater than the value of the largest access counter, the network will randomly draw the class of one of the discriminators with a score tied to be the class of the input.  

# 2.2 ClusWiSARD  

ClusWiSARD[4] is a variation of WiSARD that allows the same class to have more than one discriminator, so that sub-profiles of the same class, which do not have enough similarity between them, are learned in different places, in order to avoid saturation of the learning of a discriminator with the superposition of extremely heterogeneous patterns, but that still belong to the same class.  

ClusWiSARD is initialized with only one discriminator of each class and as new examples are learned, a verification is made to see if there is a need to create a new discriminator. In this model, the same example can be learned in more than one discriminator.  

This model can also be used for semi-supervised learning, where when a non-labeled example is submitted for learning, a classification occurs and the discriminator with the highest score will learn the example.  

# 2.3 Regression WiSARD  

An extension of WiSARD to handle prediction tasks[3, 20]. This model works with just one discriminator with each memory location having two contents/dimension (i) an access counter (same as in the WiSARD with bleaching), and; (ii) a partial prediction. During training, when a pair ¡x, $y _ { \dot { \iota } }$ is presented to the network, $\mathbf { x }$ is used to access specific memory locations and increment their access counters. The partial predictions of these same positions are incremented using the value of $y$ . At the time of prediction, when an input $\mathbf { x }$ is presented to the network, it will access the respective memory positions and Regression WiSARD will return as a prediction an average of the sum of the counters and the partial predictions accessed. This model can use different types of media. This process is showed in Fig. 2.  

# 2.4 ClusRegression WiSARD  

A variation of Regression WiSARD that is based on the same principle as ClusWiSARD of separating examples that are not sufficiently similar[3, 20]. This model is initialized with a single ReW discriminator and whenever new examples are learned there is a verification for the need to create new discriminators and a classification to determine which discriminators will learn the example. In the prediction phase, a classification is performed to determine the ReW discriminator that will perform the prediction.  

![](images/ea89133e141778ddb85f2bb06f17b0ed152a5f5884018e2191156c919e217c64.jpg)  
Figure 2: Regression WiSARD prediction[21].  

# 3 Library’s overview  

# 3.1 Implementation  

wisardpkg is hosted on GitHub at https://iazero.github.io/wisardpkg/, where users can find the latest version of the library and user documentation. Model details and features described in this publication pertain to the latest version of the model as of the date of this publication. The lib was implemented in C/C++ with a Python wrapper.  

# 3.2 Availability  

Operating systems: Linux, Mac OSX, Windows   
• Programming languages: C++ 11 and up, Python version 3.7.0 and up   
Additional system requirements: NA   
Dependency: pybind11 ( $\geq$ 2.5.0)   
List of contributors: All contributors were listed as authors with corresponding affiliations   
Language: C+ $^ +$ , with wrapper to Python 3  

Current version: 2.0.0a7  

# 3.3 Installation  

C++:  

– Clone https://github.com/IAZero/wisardpkg – Copy the wisardpkg.hpp file to the desired project – Include the library in the C++ code  

# Python:  

– Install Python PIP, if necessary – pip install wisardpkg  

To install wisardpkg in a Windows environment it is necessary to install Visual C++ additionally. To do this just download it from here and then run the installer.  

# 3.4 Architecture  

The library is divided into two main modules: models and binarization, because since these neural networks only receive binary inputs, it is necessary to treat the input to make it suitable for models. Although this is usually done through some kind of preprocessing external to wisardpkg, the library has some classes to provide support for this pipeline.  

# 3.4.1 Binarization  

All binarization classes are extensions of the BinBase class. All of them receive an array as input to their unique public method, transform, which will return a binary array. Only the public methods of each class will be described here.  

Thresholding: applies a simple threshold to a double value to generate a binary input.  

Thresholding: its only parameter is the threshold. transform  

MeanThresholding: similar to the previous one, but this time the threshold is calculated as the mean of the input data.  

MeanThresholding transform  

Thermometer: is a technique for preprocessing quantitative variables. Given a variable $d$ , a maximum value of traing test $m$ and a number of ranges $s$ , the new binary variable will have $s$ bits, with each ith bit being determined by a threshold $\begin{array} { r } { t = i * \frac { m } { s } } \end{array}$ . If $d > t$ , the $i$ th position is worth 1, otherwise 0.  

SimpleThermometer: its parameters are the thermometer size, the minimum and the maximum value in its range.   
transform  

KernelCanvas: since each WiSARD-based model is able to handle only one input size, this preprocessing[7] is capable of resizing inputs, being especially useful when dealing with time series. This uses different kernels, or divisions in the sample space of the input, replacing each value of it with the central value of the kernel where it is located.  

KernelCanvas: it is possible to instantiate it from a json file. Its parameters are the desired dimensionality and the number of kernels to be used. transform  

# 3.4.2 Models  

This module contains all the models and also the base classes from which they extend. A brief description of each sub-module and its classes follows.  

# 1. Base:  

Model: a simple trainable module – train – getsizeof  

ClassificationModel: a Model object that can calculate the similarity score in an access, as well as perform classifications  

– classify – rank – score  

RegressionModel: a Model object that performs predictions. Here the training is overwritten because of the partial prediction used in learning in this type of model.  

$$
\begin{array} { l } { { \mathrm { ~ - ~ } \mathrm { t r a i n } } } \\ { { \mathrm { ~ - ~ } \mathrm { p r e d i c t } } } \end{array}
$$  

# 2. Wisard:  

RAM: the minimal information unit in a weightless neural network, contains $2 ^ { n }$ memory positions.  

– RAM: instantiates RAM. It is possible to use a json file with RAM previously saved for this. Two additional parameters here are ignoreZero (which allows not considering the initial position of each RAM in the classification phase) and base (its default value is 2, forming the classic WiSARD for binary patterns, but when modifying it it is possible to work with patterns that use more bits and the RAM will have $b a s e ^ { n }$ memory locations).   
– getVote   
– train   
– untrain: it is possible to reverse the training process of an example, once the positions accessed are known, just subtracting their access counters.   
– getMentalImage: using the retina and the content of the RAM, it generates a representation of the learning.   
– setMapping: it is possible to choose a mapping for the RAM.  

# Discriminator:  

– Discriminator: its main parameter is the size of the tuple. It is also possible to define your mapping and instantiate it from a json file. Its main methods are: train, untrain and classify.  

Wisard: a ClassificationModel that has a set of discriminators. Its main methods are: train, untrain and classify. An optional parameter ”balanced” when set to True causes the score of each discriminator during the classification to be normalized using the number of trained examples.  

3. Cluswisard: has only one homonymous class, which will be described bellow:  

Cluswisard: its main parameters are the size of the tuple and the variables used in the verification to create new discriminators: minScore, threshold and discriminatorsLimit.   
The main methods here include train, untrain, classify, trainUnsupervised and classifyUnsupervised, the latter is applied only when it is desired to know which is the discriminator with which the example is most similar, despite classes.  

# 4. RegressionWisard:  

MeanFunctions: this module has a Mean class, which serves as the basis for several other classes that contain the methods of the means used in the prediction of Regression WiSARD (SimpleMean, PowerMean, Median, HarmonicMean, HarmonicPowerMean, GeometricMean, ExponentialMean and LogisticMean). RegressionRAM: analogous to the classification RAM, it has an extra content in its memory positions, which is the partial prediction.  

Additional parameters here include minZero and minOne, which are the minimum amount of these bits that a memory location needs to have to be considered in the prediction phase.  

RegressionWisard: The network itself, a set of RegressionRAMs. Its main parameters are the size of the tuple and the average to be used, with minZero, minOne, completeAdressSize and mapping being additional parameters. Like other Models, it can be instantiated from a json file. Its main methods are train and predict.  

5. ClusRegressionWisard: This module has only one homonymous class, which is a RegressionModel, whose main instantiation parameters are addressSize, minScore, threshold and limit. Its main methods are train and predict.  

Additionally, the library has a commons module, with exceptions and utils, and a wrapper module for Python.  

# References  

[1] I. Aleksander, W. Thomas, and P. Bowden, WISARD, a radical new step forward in image recognition, Sensor Rev., 4(3), 120-124, 1984.   
[2] Grieco, Bruno PA and Lima, Priscila MV and De Gregorio, Massimo and Fran¸ca, Felipe MG, Producing pattern examples from “mental” images, Neurocomputing, 73, 7-900, 1057–1064,2010, Elsevier.   
[3] Filho, Leopoldo AD Lusquino and Oliveira, Luiz FR and Filho, Aluizio L and Guarisa, Gabriel P and Lima, Priscila MV and Fran¸ca, Felipe MG, Prediction of Palm Oil Production with an Enhanced n-Tuple Regression Network, Proceedings of the 27th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 2019.   
[4] Douglas O. Cardoso and Danilo Carvalho and Daniel S. F. Alves and Diego F. P. de Souza and Hugo C. C. Carneiro and Carlos E. Pedreira and Priscila M. V. Lima and Felipe M. G. Fran¸ca, Financial credit analysis via a clustering weightless neural classifier, Neurocomputing,183, 70–78, 2016, doi:10.1016/j.neucom.2015.06.105.   
[5] Douglas O. Cardoso and Massimo de Gregorio and Priscila M. V. Lima and et al., A Weightless Neural Network-Based Approach for Stream Data Clustering, Intelligent Data Engineering and Automated Learning - IDEAL 2012 - 13th International Conference, Natal, Brazil, 328—335, 2012, doi:10.1007/978-3-642-32639-4.   
[6] Douglas O. Cardoso and Priscila M. V. Lima and Massimo de Gregorio and et al.,Clustering data streams with weightless neural networks, ESANN 2011, 19th European Symposium on Artificial Neural Networks, Bruges, Belgium, 2011, 201 – 206, 2-s2.0-84962014029.   
[7] de Souza, D. F. P. and Fran¸ca, F. M. G., and Lima, P. M. V., Spatio-temporal pattern classification with KernelCanvas and WiSARD,2014 Brazilian Conference on Intelligent Systems (BRACIS 2014),228–233,2014.   
[8] de Souza, D. F. P. and Fran¸ca, F. M. G. and Lima, P. M. V.,Real-time music tracking based on a weightless neural network,Proceedings of the 2015 Ninth International Conference on Complex, Intelligent, and Software Intensive Systems, 64–69,2015.   
[9] Nascimento, D. N. and de Carvalho, R. L. and Mora-Camino and F., Lima, P. M. V. and Fran¸ca, F. M. G., A WiSARD-based multi-term memory framework for online tracking of objects, Proceedings of the 23rd European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, 19–24, 2015.   
[10] Barbosa, Raul and Cardoso, Douglas O and Carvalho, Diego and Franca, Felipe MG, Weightless neuro-symbolic GPS trajectory classification,Neurocomputing,298,100– 108,2018,Elsevier.   
[11] Carneiro, H. C. C. and Fran¸ca, F. M. G. and Lima, P. M. V.,Multilingual part-of-speech tagging with weightless neural networks,Neural Networks,66,11–21,2015.   
[12] Carneiro, H. C. C. and Pedreira, C. E. and Fran¸ca, F. M. G. and Lima, P. M. V.,A universal multilingual weightless neural network tagger via quantitative linguistics,Neural Networks,91,85–101,2017.   
[13] Carneiro, Hugo CC and Pedreira, Carlos E and Franc¸a, Felipe MG and Lima, Priscila MV,The exact vc dimension of the wisard n-tuple classifier,Neural computation,31,1,176–207,2019,MIT Press.   
[14] Rangel, F., Firmino and F., Lima, P. M. V. and Oliveira, J.,Semi-Supervised Classification of Social Textual Data Using WiSARD,Proceedings of the 24th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning,165–170,2016.   
[15] de Arau´jo, Leandro Santiago and Patil, Vinay C and Prado, Charles B and Alves, Tiago AO and Marzulo, Leandro AJ and Fran¸ca, Felipe MG and Kundu, Sandip,Design of Robust, High-Entropy Strong PUFs via Weightless Neural Network,Journal of Hardware and Systems Security,3,3,235–249,2019,Springer.   
[16] Vidal, F. S. and Carneiro, H. C. C. and Rosa, P. F. F. and Fran¸ca, F. M. G., Identifica¸c˜ao de emo¸c˜oes a partir de express˜oes faciais com redes neurais sem peso,Proceedings of XI SBAI – Simp´osio Brasileiro de Automac¸˜ao Inteligente (In Portuguese),2013.   
[17] Lusquino Filho, L. A. D. and Franc¸a, F. M. G. and Lima, P. M. V.,Near-optimal facial emotion classification using WiSARD-based weightless system,Proceedings of the 26th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning,85–90,2018.   
[18] Lusquino Filho, L. A. D. and Guarisa, G. P. and Lima Filho, A. and de Oliveira, L. F. R. and Franc¸a, F. M. G. and Lima, P. M. V.,Classifying Actions Units with ClusWiSARD, Proceedings of the 28th International Conference on Artificial Neural Networks, 2019.   
[19] Lusquino Filho, L. A. D. and de Oliveira, L. F. R. and Guarisa, G. P. and Lima Filho, A. and Carneiro, H. C. C. and Franc¸a, F. M. G. and Lima, P. M. V.,A weightless regression system for predicting multi-modal empathy, Workshop Affective Behavior Analysis inthe-wild, Proc. of the 15th IEEE FG, 2020.   
[20] Lusquino Filho, L. A. D. and Oliveira, Luiz FR and Lima Filho, Aluizio and Guarisa, Gabriel P and Felix, Lucca M and Lima, Priscila MV and Franc¸a, Felipe MG,Extending the Weightless WiSARD Classifier for Regression, Neurocomputing, 2020,Elsevier.   
[21] Lusquino Filho, L. A. D.,Extending multi-label and regression capabilities of WiSARD models for multi-modal prediction,PESC/COPPE/UFRJ, 2019.  