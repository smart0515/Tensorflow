from tensorflow.keras.layers import Layer
import tensorflow as tf

# ConvLSTMCell에서 ConvLSTM의 하나의 셀에서 일어나는 연산과정을 정의합니다.
class ConvLSTMCell(Layer):
    def __init__(self, input_channels, hidden_channels, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        # Keras Layer 클래스를 상속하여 정의합니다.
      
        self.conv = tf.keras.layers.Conv2D(filters=4*hidden_channels, kernel_size=kernel_size, padding='same')
        # Conv2D 클래스의 인스턴스 생성합니다.
        # filters는 hidden_channels에 각 time step에서 4개의 게이트를 각각 계산하기 위해 *4를 합니다.

    def forward(self, inputs, state):
        h, c = state
        # state의 hidden state와 cell state tuple을 h,c 두개의 변수로 분리합니다.
    
        combined = tf.concat([inputs, h], axis=3)  
        # 입력값 inputs와 h 갑을 축 3(channel)에 따라 결합합니다.
        
        combined_conv = self.conv(combined)     
        # 결합된 값에 Conv2D 레이어를 적용합니다.
        
        cc_i, cc_f, cc_o, cc_g = tf.split(combined_conv, self.hidden_channels, axis=3)
        # 결합된 값을 축 3(channel)에 따라 4개의 채널로 분리합니다.

        i = tf.sigmoid(cc_i)     
        f = tf.sigmoid(cc_f)     
        o = tf.sigmoid(cc_o)     
        g = tf.tanh(cc_g)      
        
        c_next = f * c + i * g  
        # cell state의 다음 값을 계산합니다.
        
        h_next = o * tf.tanh(c_next) 
        # hidden state의 다음 값을 계산합니다.
        
        return h_next, (h_next, c_next)
      # 다음 hidden state값과 다음 state값을 리턴합니다.
    
# ConvLSTM의 network에 대해서 정의합니다.
class ConvLSTM(Layer):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.lstm_cells = [ConvLSTMCell(input_channels if i == 0 else hidden_channels, hidden_channels, kernel_size) for i in range(num_layers)]
        # i==0 일땐 input_channels, 나머진 hidden_channels
        # ConvLSTMCell 클래스의 인스턴스를 생성하여 lstm_cells 리스트에 추가합니다.
        
    def forward(self, inputs):
        state = (tf.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], self.hidden_channels)),
                 tf.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], self.hidden_channels)))
        
        for i, lstm_cell in enumerate(self.lstm_cells):
            inputs, state = lstm_cell(inputs, state)
            # 모델에 사용할 모든 ConvLSTM 셀을 순차적으로 호출하여 결과를 누적합니다.
            
        return state[0]
        # h_next
