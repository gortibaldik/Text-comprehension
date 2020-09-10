from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, TimeDistributed, \
                                    AdditiveAttention, Bidirectional, Add
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import EarlyStopping
from dataset_loader import Dataset
from attention import Attention

import json
import numpy as np

class SummarizationModel():
    def __init__(self,
                 use_attn=False,
                 use_bidir=False):
        self.use_attn = use_attn
        self.use_bidir = use_bidir

    def create(self,dataset):
        # creating model
        latent_dim = 500

        # Encoder:
        encoder_inputs = Input(shape=(dataset.max_len_datapoint,))
        encoder_embedding = Embedding(dataset.datapoint_vocab_size,
                                      latent_dim,
                                      trainable=True)(encoder_inputs)
        
        if self.use_bidir:
            encoder_bi_LSTM_1 = Bidirectional(LSTM(latent_dim,
                                                   return_sequences=True,
                                                   return_state=True),
                                              merge_mode="sum")
            encoder_output1, _, _, _, _ = encoder_bi_LSTM_1(encoder_embedding)
            
            encoder_bi_LSTM_2 = Bidirectional(LSTM(latent_dim,
                                                   return_sequences=True,
                                                   return_state=True),
                                              merge_mode="sum")
            encoder_output2, _, _, _, _ = encoder_bi_LSTM_2(encoder_output1)
            
            encoder_bi_LSTM_3 = Bidirectional(LSTM(latent_dim,
                                                   return_sequences=True,
                                                   return_state=True),
                                              merge_mode="sum")
            encoder_outputs, fwd_h, fwd_c, bck_h, bck_c = encoder_bi_LSTM_3(encoder_embedding)
            state_h = Add()([fwd_h, bck_h])
            state_c = Add()([fwd_c, bck_c])
          
        else:
            encoder_LSTM_1 = LSTM(latent_dim, return_sequences=True, return_state=True)
            encoder_output1, state_h1, state_c1 = encoder_LSTM_1(encoder_embedding)

            encoder_LSTM_2 = LSTM(latent_dim, return_sequences=True, return_state=True)
            encoder_output2, state_h2, state_c2 = encoder_LSTM_2(encoder_output1)

            encoder_LSTM_3 = LSTM(latent_dim, return_sequences=True, return_state=True)
            encoder_outputs, state_h, state_c = encoder_LSTM_3(encoder_output2)

        # Decoder
        decoder_inputs = Input(shape=(None,))
        decoder_embedding_layer = Embedding(dataset.label_vocab_size, latent_dim, trainable=True)
        decoder_embedding = decoder_embedding_layer(decoder_inputs)

        decoder_LSTM = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(decoder_embedding,
                                                                        initial_state=[state_h, state_c])
       
        if self.use_attn:
            attention_layer = AdditiveAttention()
            context_vector = attention_layer([decoder_outputs, encoder_outputs])
            decoder_concat_input = Concatenate()([decoder_outputs, context_vector])
            decoder_Dense = TimeDistributed(Dense(dataset.label_vocab_size, activation='softmax'))
            decoder_outputs = decoder_Dense(decoder_concat_input)

        else:
            decoder_Dense = Dense(dataset.label_vocab_size, activation='softmax')
            decoder_outputs = decoder_Dense(decoder_outputs)

        # whole model, which is used only for training, not for inference
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

        # encoder model, used only for inference with weights trained in training model
        # model without attention
        if self.use_attn:
            self.encoder = Model(encoder_inputs, [state_h, state_c, encoder_outputs])
        else:
            self.encoder = Model(encoder_inputs, [state_h, state_c])

        # decoder model, used only for inference with weights trained in training model
        decoder_embedding_inference = decoder_embedding_layer(decoder_inputs)
        decoder_input_state_h = Input(shape=(latent_dim,))
        decoder_input_state_c = Input(shape=(latent_dim,))
        dec_outputs, dec_state_h, dec_state_c = decoder_LSTM(decoder_embedding_inference,
                                                            initial_state=[decoder_input_state_h,
                                                                            decoder_input_state_c])
        if self.use_attn:
            decoder_input_hidden_states = Input(shape=(dataset.max_len_datapoint, latent_dim))
            context_inference = attention_layer([dec_outputs, decoder_input_hidden_states])
            decoder_inference_concat = Concatenate()([dec_outputs, context_inference])
            dec_outputs = decoder_Dense(decoder_inference_concat)
            self.decoder = Model([decoder_inputs,
                                  decoder_input_state_h,
                                  decoder_input_state_c,
                                  decoder_input_hidden_states], [dec_outputs,
                                                                 dec_state_h,
                                                                 dec_state_c])
        else:
            dec_outputs = decoder_Dense(dec_outputs)
            self.decoder = Model([decoder_inputs, decoder_input_state_h, decoder_input_state_c],
                                 [dec_outputs, dec_state_h, dec_state_c])

    def train(self, dataset, batch_size=64, epochs=10):    
    
        # teacher enforcing mode
        x = [dataset.datapoint_train, dataset.label_train[:, :-1]]
        # hence <BOS> tag shouldn't be in the target data
        y = dataset.label_train.reshape(dataset.label_train.shape[0],
                                        dataset.label_train.shape[1], 1)[:, 1:]

        val_x = [dataset.datapoint_test, dataset.label_test[:,:-1]]
        val_y = dataset.label_test.reshape(dataset.label_test.shape[0],
                                           dataset.label_test.shape[1], 1)[:,1:]

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        history=self.model.fit(x, 
                          y,
                          epochs=epochs,
                          batch_size=batch_size,
                          callbacks=[early_stopping],
                          validation_data=(val_x, val_y))

    def create_review(self, dataset, sequence):
        states = self.encoder.predict(sequence)

        # decoder will generate only one word which will be feeded into it in the 
        # next time step
        out_word = np.zeros((1,1))
        out_word[0,0] = dataset.label_tokenizer.word_index['start']

        review = ""
        while 1:
            predicted, h, c = self.decoder.predict([out_word]+states)
            predicted_index = int(np.argmax(predicted[0,-1,:]))
            predicted_word = dataset.label_tokenizer.index_word[predicted_index]
            
            if predicted_word != "end" and len(review) <= dataset.max_len_label:
                review += ' ' + predicted_word
            else:
                break

            if self.use_attn:
                states = [h,c] + [states[2]]
            else:
                states = [h, c]

            out_word[0,0] = predicted_index

        return review


    def save(self):

        def save_model(name, model_):
            model_.save(name+".h5")
        # saving trained model
        save_model("model", self.model)
        save_model("encoder", self.encoder)
        save_model("decoder", self.decoder)
