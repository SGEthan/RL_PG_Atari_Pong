import pickle
import numpy as np
import func


class our_pong_model:
    def __init__(self, input_size, hd_ns, decay_rate, learning_rate, from_file=False, file_name=None):
        """
        parameters:
        hd_ns: number of hidden neurons
        input_shape: tuple of the shape of the input image
        """
        if from_file:
            self.param = pickle.load(open(file_name, 'rb'))
            print('loaded model from file:', file_name)

        else:
            self.input_pixels = input_size

            # here we define our model: with two layers of their w matrix randomly initialized
            self.param = {'W1': np.random.randn(hd_ns, self.input_pixels) / np.sqrt(self.input_pixels),
                          'W2': np.random.randn(hd_ns) / np.sqrt(hd_ns)}

        # here we initialize the grad buffer to add gradients over a batch
        # and the RMSprop memory
        self.grad_buffer = {k: np.zeros_like(v) for k, v in self.param.items()}
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.param.items()}
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        This is our manually created forward function
        :param x: input data, with type of input_shape
        :return: our output, giving the probability of taking up action, and the state of our first layer
        """
        y_1 = np.dot(self.param['W1'], x)
        y_1[y_1 < 0] = 0  # Here we manually call ReLU
        y_2 = np.dot(self.param['W2'], y_1)
        p = func.sigmoid(y_2)
        return p, y_1

    def store_gradient(self, ep_y1, ep_x, ep_dy2):
        """
        This is our manually created gradient-computing function
        :param ep_y1:
        :param ep_x:
        :param ep_dy2:
        :return: None
        """
        dW2 = np.dot(ep_y1.T, ep_dy2).ravel()
        d_y1 = np.outer(ep_dy2, self.param['W2'])
        d_y1[ep_y1 <= 0] = 0  # bp PReLU
        dW1 = np.dot(d_y1.T, ep_x)

        self.grad_buffer['W1'] += dW1
        self.grad_buffer['W2'] += dW2

    def back_propagation(self):
        for k, v in self.param.items():
            grad = self.grad_buffer[k]
            self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * grad ** 2
            self.param[k] += self.learning_rate * grad / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
            self.grad_buffer[k] = np.zeros_like(v)

    def save_model(self, path):
        pickle.dump(self.param, open(path, 'wb'))
