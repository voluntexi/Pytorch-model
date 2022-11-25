# This is a simple demo for using RNN to predict the phrases sentiment in the dataset.
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dataset import test_data, train_data
from numpy import random

'''
h=[64,1]
why[2,64]
whh[64,64]
wxh[2,64]
bh[64,1]
by[2,1]
y=[2,1]
'''


class RNN:
    def __init__(self, input_size, output_size, hidden_size=64):
        # Define that hidden layer have 64 neurons
        self.Whh = random.randn(hidden_size, hidden_size) / 1000
        self.Wxh = random.randn(hidden_size, input_size) / 1000
        self.Why = random.randn(output_size, hidden_size) / 1000
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.last_hs = {}
        self.last_inputs = []

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        self.last_inputs = inputs
        self.last_hs = {0: h}
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h
        # enumerate:枚举  @:矩阵的乘法
        y = self.Why @ h + self.by
        return y

    def backprop(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)
        d_Why = d_y @ self.last_hs[n].T
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)
        d_by = d_y
        # Calculate dL/dh for the last h.
        d_h = self.Why.T @ d_y
        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
            d_bh += temp
            d_Whh += temp @ self.last_hs[t].T
            d_Wxh += temp @ self.last_inputs[t].T
            d_h = self.Whh @ temp
        # Make the values between -1 and 1
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)
        # Using gradient descent to update the weights and biases
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


def softMax(x):
    return np.exp(x) / sum(np.exp(x))


def createInputs(text):
    # Returns an array of one-hot vectors representing the words in the input text

    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs


def processData(data, backprop=True):
    items = list(data.items())
    random.shuffle(items)  # 打乱顺序
    loss = 0
    num_correct = 0
    for x, y in items:
        inputs = createInputs(x)
        target = int(y)
        out = rnn.forward(inputs)
        prob = softMax(out)
        loss += (-np.log(prob[target]))
        num_correct += int(np.argmax(prob) == target)  # argmax 最大值的索引
        if backprop:
            # get DL\Dy
            d_y = prob
            d_y[target] -= 1

            rnn.backprop(d_y)
    return loss / len(data), num_correct / len(data)


if __name__ == '__main__':
    # Generate "one-hot" word vector
    vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
    vocab_size = len(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}
    # Generate rnn
    rnn = RNN(input_size=len(vocab), output_size=2)
    print("负情感" if np.argmax(softMax(rnn.forward(createInputs("i am not earlier")))) == 0 else "正情感")
    for epoch in range(1000):
        train_loss, train_acc = processData(train_data)
        if epoch % 100 == 99:
            print(f'--- Epoch {epoch + 1}')
            print(f'Train Loss: {np.round(train_loss, 3)} | Accuracy: {np.round(train_acc, 3)}')
            test_loss, test_acc = processData(test_data, backprop=False)
            print(f'Test Loss: {np.round(test_loss, 3)} | Accuracy: {np.round(test_acc, 3)}')
    print("负情感"if np.argmax(softMax(rnn.forward(createInputs("i am not earlier"))))==0 else "正情感")
