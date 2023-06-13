import numpy as np

class StochasticGradientDescent:

    def __init__(self, input_size, output_size, learning_rate=0.01, num_epochs=1000, batch_size=32):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weights = None
        self.biases = None
    
    def fit(self, training_data):
        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
        self.biases = np.zeros(self.output_size)
        
        for epoch in range(self.num_epochs):
            np.random.shuffle(training_data)
            
            for i in range(0, len(training_data), self.batch_size):
                batch = training_data[i:i+self.batch_size]
                
                inputs = batch[:, :-1]
                targets = batch[:, -1]
                outputs = np.dot(inputs, self.weights) + self.biases
                
                loss = self._compute_loss(outputs, targets)
                
                grad_outputs = self._compute_gradient(inputs, outputs, targets)
                grad_weights = np.dot(inputs.T, grad_outputs) / self.batch_size
                grad_biases = np.mean(grad_outputs, axis=0)
                
                self.weights -= self.learning_rate * grad_weights
                self.biases -= self.learning_rate * grad_biases
                
            print(f"Epoch {epoch+1} Loss: {loss}")
    
    def predict(self, inputs):
        if self.weights is None or self.biases is None:
            raise RuntimeError("Model has not been trained yet.")
        return np.dot(inputs, self.weights) + self.biases
    
    def _compute_loss(self, outputs, targets):
        loss = np.mean((outputs - targets) ** 2)
        return loss

    def _compute_gradient(self, inputs, outputs, targets):
        # Compute the gradient of the mean squared error loss

        # Calculate the difference between outputs and targets
        diff = outputs - targets

        # Reshape grad_outputs to match the shape of inputs
        grad_outputs = np.reshape(diff, (len(inputs), 1))

        # Compute the gradient of the weights
        grad_weights = np.dot(inputs.T, grad_outputs) / self.batch_size

        # Compute the gradient of the biases
        grad_biases = np.mean(grad_outputs, axis=0)

        # Return the gradients
        return grad_weights, grad_biases


