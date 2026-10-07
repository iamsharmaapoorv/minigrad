import random
from value import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    def __init__(self, no_w=3, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(no_w)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return activation.relu() if self.nonlin else activation

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else "linear"} Neuron ({len(self.w)})"


class Layer(Module):
    def __init__(self, no_input, no_neuron, nonlin=True) -> None:
        self.neurons = [Neuron(no_input, nonlin) for _ in range(no_neuron)]
        self.nonlin = nonlin

    def __call__(self, x):
        activations = [neuron(x) for neuron in self.neurons]
        activations = [
            activation.relu() if self.nonlin else activation for activation in activations]
        return activations[0] if len(activations) == 1 else activations

    def parameters(self):
        parameters = []
        for neuron in self.neurons:
            parameters.extend(neuron.parameters())
        return parameters

    def __repr__(self) -> str:
        ans = ""
        for neuron in self.neurons:
            ans += neuron.__repr__()
            ans += "\n"
        return ans


class MLP(Module):
    def __init__(self, nin, nouts) -> None:
        inputs = [nin]+nouts
        self.layers = [Layer(inputs[i], inputs[i+1], i != len(nouts)-1)
                       for i in range(len(inputs)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters

    def __repr__(self) -> str:
        ans = "MLP\n"
        for i in range(len(self.layers)):
            ans += f"Layer {i}\n"
            ans += self.layers[i].__repr__()
        return ans
