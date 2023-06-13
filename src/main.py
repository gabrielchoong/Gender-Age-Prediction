import numpy as np
from sgd import StochasticGradientDescent

# Create an instance of StochasticGradientDescent
sgd = StochasticGradientDescent(input_size=2, output_size=1, learning_rate=0.01, num_epochs=1000, batch_size=32)

# Prepare training data
training_data = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
])

# Train the model
sgd.fit(training_data)

# Make predictions
inputs = np.array([[0.5, 0.2], [0.1, 0.7]])
predictions = sgd.predict(inputs)

print("Predictions:", predictions)