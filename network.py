# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import json

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):  # sizes是一个列表，表示每一层的神经元数量。
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)    # 首先，通过计算sizes列表的长度来确定网络的层数，并将结果存储在实例变量num_layers中。
        self.sizes = sizes  # 然后，将传入的sizes列表赋值给实例变量self.sizes，以便稍后引用。

        # 接下来，为每一层（从第二层开始，因为第一个层被视为输入层，不需要偏置项）初始化偏置向量。
        # 这里使用了列表推导式，通过遍历sizes[1:]（即从索引1开始到最后一个元素），对于每个元素y，生成一个大小为(y, 1)的随机数数组，
        # 这些随机数遵循标准正态分布（均值为0，标准差为1）。这些随机数数组被存储在self.biases列表中，每个元素对应网络的一个隐藏层或输出层的偏置向量。
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]   

        # 同样地，为每一层（除了最后一层，因为输出层不需要权重矩阵）的连接权重进行初始化。
        # 这里使用了zip函数来同时遍历sizes[:-1]（即除最后一个元素外的所有元素）和sizes[1:]之间的配对。
        # 对于每对(x, y)，生成一个大小为(y, x)的随机数矩阵，这些随机数也遵循标准正态分布。
        # 这些随机数矩阵被存储在self.weights列表中，每个元素对应网络的一个隐藏层与前一层之间的权重矩阵。
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):

        # 实现了神经网络的前向传播过程。
        # 在每一层，它都会计算当前层的输入（即上一层的输出），并通过加上偏置向量后应用sigmoid激活函数来得到当前层的输出。
        # 这个过程会一直持续到最后一层，然后返回最后一层的输出作为整个网络的输出。
        """Return the output of the network if ``a`` is input."""

        # 这行代码使用zip函数来同时迭代self.biases和self.weights两个列表。
        # 这意味着我们将逐层处理网络中的每一层，从输入层到输出层。
        for b, w in zip(self.biases, self.weights):

            # 这行代码执行了当前层的计算。首先，使用np.dot(w, a)计算当前层的权重矩阵w与上一层的激活值a的点积。
            # 然后，加上当前层的偏置向量b。
            # 最后，应用sigmoid激活函数对加法结果进行非线性转换。
            # sigmoid函数将任意实数映射到(0, 1)范围内，使得输出可以被解释为概率。
            a = sigmoid(np.dot(w, a)+b)

            # 最后，返回经过所有层计算后的最终输出。
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        # SGD方法实现了使用小批量随机梯度下降训练神经网络的过程。
        # 它通过分割训练数据为多个小批量，在每个小批量上更新网络的权重和偏置，以此来优化网络的性能。


        # training_data：一个由元组(x, y)组成的列表，代表训练输入和期望的输出。
        # epochs：训练过程中要进行的周期次数。
        # mini_batch_size：每次更新权重时使用的小批量大小。
        # eta：学习率，控制权重更新的步长。
        # test_data（可选）：如果提供，则在每个周期结束后评估网络对测试数据的表现。
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        best_accuracy =0
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]  # 每隔 mini_batch_size 个数字取一次。
            # 遍历每个小批量，对其执行权重更新
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                n_evaluate=self.evaluate(test_data)
                current_accuracy =round(n_evaluate/n_test*100,2)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    self.save('mynet/mynet_{}'.format(current_accuracy))
                print("Epoch {} : {} / {}  {}%  {}%".format(j,n_evaluate,n_test,current_accuracy,best_accuracy))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # 对当前小批量中的每一对输入和期望输出调用backprop方法，该方法执行反向传播计算，并返回偏置和权重的梯度。
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 累加每次反向传播计算得到的梯度。
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_data = list(test_data)
        n_test = len(test_data)
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        n_evaluate=sum(int(x == y) for (x, y) in test_results)
        current_accuracy =round(n_evaluate/n_test*100,2)
        print(test_results)
        print("{} / {}  {}%".format(n_evaluate,n_test,current_accuracy))

        return n_evaluate

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    
    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#### Loading a Network
def load_net(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net



# net = Network([2, 3, 1])

# print(net.sizes)
# print(net.biases[0])
# print(net.weights[0])


# print('======')
# print(net.biases[1])
# print(net.weights[1])