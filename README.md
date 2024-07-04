PLANT DISEASE PREDICTION DL PROJECT
This repository is about building an Image classifier using CNN and Multilayer perceptron with Python on Plant Disease Prediction.

KAGGLE DATASET LINK : https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

The dataset has 38 classes and colored, segmented and grey-scale images. In this project I have only used colored images and 5 epochs for both models to make fair comparision.

EXPLANATION :

Multilayer Perceptron (MLP) Implementation:
The MLP code uses the Keras library to define and train a neural network for a classification task. Here are the key components and steps:

Model Definition: The model is defined using the Sequential API, which allows stacking layers one after the other.

Input layer: Flattens the input image data to a 1D array.
Dense layers with ReLU activation: These layers perform the actual computations, transforming the input data through linear operations and activation functions.
Dropout layers: Regularization technique that helps prevent overfitting by randomly deactivating a fraction of neurons during training.
Output layer: Uses the softmax activation function for multi-class classification, outputting probabilities for each class.
Compilation: The model is compiled with the Adam optimizer, categorical cross-entropy loss function (suitable for multi-class classification), and accuracy as the metric for evaluation.

Training: The model is trained using the fit() method, where training data is provided along with parameters like batch size, epochs, and validation data for monitoring performance.

Model Evaluation: After training, the model's performance is evaluated on validation data to assess its accuracy.

Convolutional Neural Network (CNN) Implementation:
The CNN code utilizes the Sequential API in Keras to build and train a CNN model for image classification. Here's a breakdown of the key steps:

Model Architecture: The model is defined with convolutional layers followed by max-pooling layers to extract features from input images.

Conv2D layers: These perform convolution operations to extract features, using ReLU activation for non-linearity.
MaxPooling2D layers: Downsampling layers that reduce spatial dimensions, retaining important information.
Flatten layer: Converts the 2D feature maps to a 1D vector for input to the dense layers.
Dense layers: Fully connected layers with ReLU activation for further processing.
Output layer: Uses softmax activation for multi-class classification.
Compilation: Similar to MLP, the CNN model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.

Training: The model is trained using fit() with training data and validation data generators. Parameters like steps per epoch, epochs, and validation steps are set.

Model Evaluation: After training, the model's performance is evaluated on validation data to measure accuracy.

Overall, both implementations aim to create and train neural network models for image classification tasks, with the CNN specifically designed for handling image data by leveraging convolutional and pooling layers to extract meaningful features.

REQUIREMENTS Python 3.x TensorFlow Keras Matplotlib (for plotting)

USAGE Clone the repository:

bash Copy code git clone https://github.com/your-username/image-classification-cnn.git Install dependencies:

bash Copy code pip install -r requirements.txt Train the model:

bash Copy code python train_model.py Evaluate the model:

bash Copy code python evaluate_model.py Use the predictive system:

bash Copy code python predict_image.py /path/to/image.jpg Replace /path/to/image.jpg with the path to the image you want to classify.
