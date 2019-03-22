# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/class_distribution.png "distribution"
[image2]: ./plots/RGBvsGREY.png "Grayscaling"
[image3]: ./german_traffic_signs/random_noise.jpg "Random Noise"
[image4]: ./german_traffic_signs/30.jpg "Traffic Sign 1"
[image5]: ./german_traffic_signs/30-nicht.jpg "Traffic Sign 2"
[image6]: ./german_traffic_signs/Priority.jpg "Traffic Sign 3"
[image7]: ./german_traffic_signs/sign-1418256__340.jpg "Traffic Sign 4"
[image8]: ./german_traffic_signs/Vorfahrt-achten.jpg "Traffic Sign 5"
[image9]: ./plots/learning_curve.png "Learning Curve"
[image10]: ./plots/Online_images.png "Online images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data distributed in 43 difference class. The data set set highly un balanced, so data augmention might help to improve the model accuracy. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because traffic signs are distinguishable based on the shapes of signs, regardless of colors.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because having a wider distribution in the training data would make it difficult to train. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grey image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x16 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 14x14x80  	|
| RELU					|												|
| Dropout				| probability 0.5								|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x30  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x30   				|
| Fully connected		| inputs 1080, output 750						|
| RELU					|												|
| Dropout				| probability 0.5								|
| Fully connected		| inputs 750, output 350		  			    |
| RELU					|												|
| Fully connected		| inputs 350, output 43  						|
| Softmax				|           									|


 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, because it combining the advantages of two other extensions of stochastic gradient descent optimazation methods, Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp). I choose batch size of 128 after trying 32, 64, which didn't lose any accury but speed up training. I use exponently decaying learning rate starting from 0.001 decay 2% every 10 epochs.  


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I trained the model 200 epochs then chose trained result at epoch 130 according to learning curve.
![alt text][image9]
My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.6% 
* test set accuracy of 96.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?

I use Lenet architecute.
* Why did you believe it would be relevant to the traffic sign application?

Proformance of Lenet architechure was proven with a lot of similar data set and can be easly modified according to complexity of the problem.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Model achivied 96.4% accuracy on test data set while having 100% and 97.6% accuracy on training and validation data set respectly. This is resonal outcome for this type of problem. Accurcay on validation and testing data set can be improved further with better fine tuning.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image10]

The first image might be difficult to classify because there 8 different End of speed limit sign for different speed. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (60km/h) 							| 
| Priority road			| Priority road									|
| Road work 			| Road work  	    							|
| Children crossing		| Children crossing								|
| Yield					| Yield											|



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of orginal data set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .49         			| Speed limit (60km/h)   						| 
| .35     				| Speed limit (50km/h) 							|
| .04					| Keep left										|
| .02	      			| Children crossing				 				|
| .02				    | No entry      		     					|



For the other 4 images, model predict correctly with probability of 1.0.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


