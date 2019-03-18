# 
# # First we define a model using keras/kraino
# from keras.layers.core import Activation
# from keras.layers.core import Dense
# from keras.layers.core import Dropout
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import GRU
# from keras.layers.recurrent import LSTM
#
# from kraino.core.model_zoo import AbstractSequentialModel
# from kraino.core.model_zoo import AbstractSingleAnswer
# from kraino.core.model_zoo import AbstractSequentialMultiplewordAnswer
# from kraino.core.model_zoo import Config
# from kraino.core.keras_extensions import DropMask
# from kraino.core.keras_extensions import LambdaWithMask
# from kraino.core.keras_extensions import time_distributed_masked_ave
#
# # This model inherits from AbstractSingleAnswer, and so it produces single answer words
# # To use multiple answer words, you need to inherit from AbstractSequentialMultiplewordAnswer
# class BlindRNN(AbstractSequentialModel, AbstractSingleAnswer):
#     """
#     RNN Language only model that produces single word answers.
#     """
#     def create(self):
#         self.add(Embedding(
#                 self._config.input_dim,
#                 self._config.textual_embedding_dim,
#                 mask_zero=True))
#         #TODO: Replace averaging with RNN (you can choose between LSTM and GRU)
# #         self.add(LambdaWithMask(time_distributed_masked_ave, output_shape=[self.output_shape[2]]))
#         self.add(LSTM(self._config.hidden_state_dim,
#                       return_sequences=False))
#         self.add(Dropout(0.5))
#         self.add(Dense(self._config.output_dim))
#         self.add(Activation('softmax'))

# to make this work Kraino folder needs to be downloaded from the corresponding github
