from collections import deque 

class Value:
    def __init__(self, data, prev = set(), op="") -> None:
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self.op = op
        self.prev = prev

    # dy/dx = (f(x+h) - f(x))/h

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward

        return output

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward

        return output

    def __pow__(self, other):
        output = Value(self.data ** other, (self,), "**")

        def _backward():
            self.grad += (other * self.data**(other-1)) * output.grad
        output._backward=_backward
        return output

    def __rmul__(self, other ):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self*(other**-1)

    
    def backward(self):
        visited = set()
        order = []
        def dfs(node):
            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    dfs(child)
                order.append(node)

        dfs(self)
        self.grad = 1
        for elem in reversed(order):
            elem._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"