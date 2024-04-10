**A simple CNN (Convolutional Neural Network) built from scratch using Java (plus some UI).**

**Dataset**: MNIST - hand-written numbers (Training set: 60000, Test set: 10000).
I used the CSV version of the dataset: https://pjreddie.com/projects/mnist-in-csv/

Implemented batch training because the training speed and training accuracy was too low on the whole dataset and I didn't know how to use GPU on Java.

Accuracy after 10 epochs was **0.85** with batch size of 6000 and a simple architecture: CONV, POOL, FULLYCONNECTED (reLU).

**TO BUILD**: just run the main.java.

**References**: 
- Rea is Online: https://www.youtube.com/watch?v=3MMonOWGe0M&list=PLpcNcOt2pg8k_YsrMjSwVdy3GX-rc_ZgN
- Victorzhou: https://victorzhou.com/blog/intro-to-cnns-part-1/

