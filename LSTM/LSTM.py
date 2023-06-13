import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 각 게이트(input, forget, output)에 대한 가중치와 bias 초기화
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) # forget 게이트 가중치
        self.bf = np.zeros((hidden_size, 1)) # forget 게이트 bias
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) # input 게이트 가중치
        self.bi = np.zeros((hidden_size, 1)) # input 게이트 bias
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) # output 게이트 가중치
        self.bo = np.zeros((hidden_size, 1)) # output 게이트 bias
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) # candidate 값 가중치
        self.bc = np.zeros((hidden_size, 1)) # candidate 값 bias

    def sigmoid(self, x):
        # sigmoid 함수 구현
        return 1 / (1 + np.exp(-x))

    def forward(self, x, h_prev, c_prev):
        # 입력값과 이전 hidden state, cell state 연결
        x = np.column_stack((h_prev, x))

        # forget gate 계산
        f = self.sigmoid(np.dot(self.Wf, x) + self.bf)
        # input gate 계산
        i = self.sigmoid(np.dot(self.Wi, x) + self.bi)
        # candidate 값 계산
        c_cand = np.tanh(np.dot(self.Wc, x) + self.bc)
        # cell state 계산
        c = f * c_prev + i * c_cand
        # output gate
