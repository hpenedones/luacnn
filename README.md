luacnn
======

Simple example of how to use torch7 convolutional neural networks for hand digit recognition, on the usps dataset. 

#### Requirements

* Torch7 machine learning library (www.torch.ch)
* image torch package 
* qtlua torch package (if you want to visualize the data)


#### Description

Each image contains a total 1100 examples of that digit (organized in 33 columns and 34 rows). 
Each example is 16x16 pixel. We will use 1000 samples for training and the remaining for testing.

The first layer of the network is a set of local filters that are applied convolutionally across the image. 
This is followed by a sub-sampling layer to reduce data dimensionality and introduce a bit of translation invariance. 
After that, a non-linear transfer function (hyperbolic tangent), which keeps the responses of that layer bounded to [-1,1]. 
We then have a linear layer with 10 outputs (one for each digit). 
Finally, a SoftMax operation keeps values in [0,1] and the sum over all classes is 1, so these can be interpreted as probabilities (of data belonging to that class). 
Here we actually use a LogSoftMax which gives log-probabilities.


#### Classification performance

In this setup, I obtained test errors around 3% or 4%. 
Note that training the network with stochastic gradient descent will give you slightly different results each time you do it (because there is randomness selecting the next training sample).

#### Possible improvements

I tried to keep this example as simple as possible, but one can expand it in several directions:

##### Speeding-up training:

  *  Normalize data before training
  *  Learning rate dependent on number of neurons in layer

##### Model selection

  *  Use a validation dataset to avoid over-fitting

##### Visualization

  *  Show current error during training
  *  Display wrong classified examples at test time   

##### Multi-task

   * Imagine you want now to recognize letters. One can jointly train two networks (one for digits and another for letters) and share the first layer parameters of the networks. This is really easy in torch, using the clone and share methods.  


#### Credits

David Grangier, Ronan Collobert (mentoring/collaboration during my 2010 Summer internship at NEC Labs, Princeton, US and at Idiap Research Institute, Switzerland).


#### Recommended reading

Y. LeCun, L. Bottou, G. Orr and K. Muller: Efficient BackProp, in Orr, G. and Muller K. (Eds), Neural Networks: Tricks of the trade, Springer, 1998 (download pdf)



