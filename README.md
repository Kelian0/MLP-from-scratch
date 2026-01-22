# MLP From Scratch

A lightweight, object-oriented implementation of a Multi-Layer Perceptron (MLP) built entirely from scratch using Python and NumPy. This project demonstrates the inner workings of neural networks, including forward propagation, backpropagation, and gradient descent, without relying on deep learning frameworks like TensorFlow or PyTorch.

## Project Structure

* **`mlp.py`**: Contains the main `MLP` class which orchestrates the network creation, training loop, and prediction.
* **`Unit.py`**: Defines the building blocks of the network:
* `InputUnit`: Handles input features.
* `NeuralUnit`: Represents a single neuron (weights, bias, Sigmoid activation, gradient calculation).
* `Loss`: Handles the binary cross-entropy loss calculation.


* **`Experiment.ipynb`**: A Jupyter Notebook demonstrating how to use the library to solve a binary classification problem (using the `make_circles` dataset).


## Usage

Here is a simple example of how to initialize, train, and use the MLP for a binary classification task.

### 1. Import and Prepare Data

```python
import numpy as np
from sklearn.datasets import make_circles
from mlp import MLP

X, y = make_circles(n_samples=300, factor=0.1, noise=0.2, random_state=5)

```

### 2. Define Architecture and Initialize

The architecture is defined as a list of integers representing the number of neurons in each layer (be aware that the loss layer is not included).

```python
archi = [2, 10, 1]

model = MLP(X, y, archi, seed=5)

model.visualize()

```

### 3. Train the Model

Train the model using Stochastic Gradient Descent.

```python
epochs = 50
learning_rate = 0.01

model.train(epochs, learning_rate)
```

## How it Works

1. **Initialization**: The network is built layer by layer. `NeuralUnit`s are plugged into the preceding layer's units.
2. **Forward Pass**: Data flows from `InputUnit`s through `NeuralUnit`s (performing ) to the `Loss` unit.
3. **Loss Calculation**: The network uses a Log Loss (Binary Cross-Entropy) logic for binary classification.
4. **Backpropagation**: Error terms (`deltas`) are propagated backward from the Loss unit through the network using the chain rule.
5. **Update**: Weights and biases are updated using the calculated gradients and the learning rate ().

## Example Results

Using the `Experiment.ipynb`, the network successfully separates non-linearly separable data (concentric circles) after training.

*(You can include the plot generated in cell 6 of your notebook here if you have the image)*

## License

This project is open-source. Feel free to use it for educational purposes.