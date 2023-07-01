# Distributed-Denial-of-service-DDoS


//SOFTWARE REQUIREMENTS
Language                : Python
IDE       	       : Anaconda Spyder
Libraries	       : Pandas, NumPy, TensorFlow, etc 

//INSTALL
pip install tensorflow==2.5.0
pip install keras==2.3.1


//INTRODUCTION

The increase in DDoS attacks clearly reveal that gaps still exist on how DDoS attack is efficiently detected, analyzed, and mitigated in a timely manner to ensure network service availability. They mainly aim to system resources and network bandwidth, ranging from Network layer to Application layer. Since the first DDoS attack occurred in 1999, DDoS has become a critical, widespread, and rapidly evolving threat in the world. According to a survey from Radware, DDoS is currently the largest threat (50% respondents in the survey) for organizations. Currently, main attack vectors include UDP flood, HTTP flood, SYN flood etc and pose serious threats to both systems and networks. However, it is hard to detect DDoS attack automatically because in most cases attack traffic is very similar to legitimate traffic and attackers try to mimic flash crowds.  An attack activity with insufficient traffic can even be seen as a legitimate one in early stages. Many researchers try statistical machine learning methods to identify DDoS attacks. We propose a new detection and family classification approach based on a set of network flow features.

//PROPOSED METHOD

1. A DDoS attack proposed method using CNN (Convolutional Neural Networks) is a type of machine learning technique used to detect malicious traffic and identify potential 
   DDoS attacks. 
2. It uses deep learning algorithms to analyze incoming traffic and identify suspicious activity.
3. It is able to learn patterns in the data and detect any anomalies that could indicate an attack. 
4.The method has been successfully used to detect both UDP and TCP floods.

//OBJECTIVES 

Use LSTM RNN technique and TensorFlow API's to produce the code for a tensor flow AI model capable of detecting DDoS flooding attacks. Determine the possible fine-tune evaluation parameters required by the AI model to achieve the best detection accuracy and discard attacks within the shortest time. Determine the performance of the AI model when trained on CPU based systems.

//MOTIVATION

Detection of DDoS flooding attacks particularly UDP, TCP and ICMP flooding attack types. 
Achieving a higher detection accuracy and less false alarm rate.

//APPLICATIONS

This Project is an ideal platform for implementing projects involving distributed applications, security solutions, and decentralized network administration.

//IMPLEMENTATION MODULES

In the Existing, Machine learning methods identify DDoS attacks based on statistical features and perform better than statistical ways. 
However, they are prone to several drawbacks: 
Requiring extensive network expertise and experiments in DDoS to select       proper statistical features;
Limited to only one or several DDoS attack vectors;
Requiring updating its model and threshold value to satisfy the changes of systems and attack vectors;
Vulnerable to slow attack rate.

//RESULTS

1. Detecting DDoS flooding attacks on a network.
2. Classifying the attack without disturbing the network.

//CONCLUSION

In this project, we proposed a deep learning based DDoS detection approach, Deep LSTM Defence. It helps improve the performance of identifying DDoS attack traffic. 
We formulate the DDoS detection as a sequence classification problem and transform the packet-based DDoS detection to the window based detection. Deep LSTM Defence approach consists of CNN, RNN (LSTM, GRU) and fully connected layers. The experimental results demonstrate that Deep LSTM Defence reduces error with conventional machine learning method. Recurrent Neural Network can learn much longer historical features than conventional machine learning methods.
