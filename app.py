from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler


class SingleLayerSVMNN:
    def __init__(self):
        self.Weights = None
        self.bias = None
        self.LR = None
        self.num_iters = None
        self.X = None
        self.Y = None
    
    def forward_pass(self):
        logits = np.dot(self.X,self.Weights) + self.bias
        return logits
    
    def SVMLossWithDerivative(self, logits):
        #print(logits[np.arange(logits.shape[0]), self.Y])
        correct_scores = logits[np.arange(logits.shape[0]), self.Y].reshape(-1, 1)
        diff = np.maximum(0, logits - correct_scores + 1)
        diff[np.arange(logits.shape[0]), self.Y] = 0
        #print(diff)
        
        loss = np.sum(diff) / logits.shape[0] 
        
        diff[diff > 0] = 1
        row_sum = np.sum(diff, axis=1)
        diff[np.arange(logits.shape[0]), self.Y] = -row_sum
        
        return loss, diff

    def GetDerivative(self,LossDerivative):
        dW = np.dot(self.X.T, LossDerivative)
        db = np.sum(LossDerivative, axis=0, keepdims=True)
        return [dW,db]

    def GradientDescent(self,LossDerivative):
        derivative = self.GetDerivative(LossDerivative)
        new_weights = self.Weights - (derivative[0] * self.LR)
        new_bias = self.bias - (derivative[1] * self.LR)
        return new_weights,new_bias

    def SaveModel(self, filepath='svm_model.npz'):
        np.savez(filepath, Weights=self.Weights, bias=self.bias)
        print(f"Model saved to {filepath}")
    
    def LoadModel(self, filepath='svm_model.npz'):
        data = np.load(filepath)
        self.Weights = data['Weights']
        self.bias = data['bias']
        print(f"Model loaded from {filepath}")


    def Train(self,X, Y, weights = None, bias = None, printer=10, alpha=0.01, max_iterations=40):
        self.X = X
        self.Y = Y
        self.LR = alpha
        self.num_iters = max_iterations

        num_classes = np.max(Y) + 1
        print(num_classes)
        num_features = X.shape[1]
        num_train = X.shape[0]
        
        if weights == None:
            self.Weights = 0.01 * np.random.randn(num_features, num_classes)
        else:
            self.Weights = weights
        
        if bias == None:
            self.bias = np.random.randn(1, num_classes)
        else:
            self.bias = bias

        print("\n##################################################################")
        print("Initial Weights = ",self.Weights)
        print("Initial Bias = ",self.bias)
        print("##################################################################\n")

        for i in range(self.num_iters):
            # Forward pass
            logits = self.forward_pass()
            # Compute loss and gradients
            loss, derivative = self.SVMLossWithDerivative(logits)
            self.Weights, self.bias = self.GradientDescent(derivative)
            if i % printer == 0:
                print("iteration {}: loss {}".format(i, loss))


# Assuming SingleLayerSVMNN is already defined in the same script or imported
app = Flask(__name__)

# Load the trained model
svm = SingleLayerSVMNN()
svm.LoadModel(filepath = "svm_model.npz")


# Load the saved scaler
scaler = load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        feature_1 = float(request.form['feature_1'])
        feature_2 = float(request.form['feature_2'])
        feature_3 = float(request.form['feature_3'])
        feature_4 = float(request.form['feature_4'])

        # Create input array and reshape for a single sample
        features = np.array([[feature_1, feature_2, feature_3, feature_4]])
        
        # Apply the same scaling as in training
        features_scaled = scaler.transform(features)

        # Set model input and make prediction
        svm.X = features_scaled
        logits = svm.forward_pass()
        prediction = np.argmax(logits, axis=1)[0]
        
        # Map prediction to class names (for Iris dataset)
        class_names = ['Setosa', 'Versicolour', 'Virginica']
        predicted_class = class_names[prediction]
        
        return render_template('index.html', prediction=f'Predicted class: {predicted_class}')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
