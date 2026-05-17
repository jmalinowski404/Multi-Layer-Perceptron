import random;
from functions import *;

class Neuron:
    def __init__(self, num_inputs, use_bias=False):
        self.weights = [random.uniform(-0.5, 0.5) for i in range(num_inputs)]
        self.use_bias = use_bias;
        
        self.bias_weight = random.uniform(-0.5, 0.5) if use_bias else 0.0;
        
        self.output = 0.0;
        self.delta = 0.0;
        self.prev_dweight = [0.0] * num_inputs;
        self.prev_dbias = 0.0;
        
class Layer:
    def __init__(self, num_neurons, num_inputs, use_bias=False):
        neurons_arr = []
        
        for i in range(num_neurons):
            neurons_arr.append(Neuron(num_inputs, use_bias));
            
        self.neurons = neurons_arr;
        
class MLP:
    def __init__(self, architecture, use_bias=False, activation_func=sigmoidalUnipolar):
        layers_arr = []
        
        for i in range(1, len(architecture)):
            num_inputs = architecture[i - 1]
            num_neurons = architecture[i]
            layers_arr.append(Layer(num_neurons, num_inputs, use_bias))
            
        self.layers = layers_arr;
        self.use_bias = use_bias;
        self.activation_func = activation_func;
        
    def forward(self, inputs):
        current_inputs = inputs;
        
        for layer in self.layers:
            next_inputs = []
            
            for neuron in layer.neurons:
                weighted_sum = 0;

                for i in range(len(current_inputs)):
                    weighted_sum += current_inputs[i] * neuron.weights[i]
                    
                if (self.use_bias): weighted_sum += 1.0 * neuron.bias_weight
                
                neuron.output = self.activation_func(weighted_sum);
                
                next_inputs.append(neuron.output)
            
            current_inputs = next_inputs
        
        return current_inputs
        
    def backward(self, inputs, expected, lr, momentum):
        output_layer = self.layers[-1]
        
        for i, neuron in enumerate(output_layer.neurons):
            error = expected[i] - neuron.output
            neuron.delta = error * neuron.output * (1 - neuron.output)
            
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            for j, neuron in enumerate(current_layer.neurons):
                error = 0.0
                for next_neuron in next_layer.neurons:
                    error += next_neuron.delta * next_neuron.weights[j]
                neuron.delta = error * neuron.output * (1 - neuron.output)
                
        for layer_idx, layer in enumerate(self.layers):
            layer_inputs = inputs if layer_idx == 0 else [n.output for n in self.layers[layer_idx - 1].neurons]
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    dw = lr * neuron.delta * layer_inputs[i] + momentum * neuron.prev_dweight[i]
                    neuron.weights[i] += dw
                    neuron.prev_dweight[i] = dw
                if self.use_bias:
                    db = lr * neuron.delta * 1.0 + momentum * neuron.prev_dbias
                    neuron.bias_weight += db
                    neuron.prev_dbias = db