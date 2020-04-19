import gensim
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import csv
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.stem import PorterStemmer
import numpy as np
import os
from gensim.models import Word2Vec
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from attention_layer_script import AttentionLayer
from tensorflow.keras.callbacks import EarlyStopping
import string


DATAST_FILE_PATH = r"C:\Users\device_lab\Desktop\After_Parser\Reviews.csv"
OUTPUT_PATH = ''
LOAD_PRETRAINED_MODEL = False


class LSTM_Model:
    def __init__(self,inputs_length,outputs_length):
        from keras import backend as K 
        K.clear_session()

        #if not load_model:
        #latent_dim = LSTM layer num nodes
        latent_dim = 300
        embedding_dim=100
        max_text_len = 500
        self.embedding_dim = embedding_dim
        self.max_text_len = max_text_len
        self.latent_dim = latent_dim

        # Encoder
        encoder_inputs = Input(shape=(max_text_len,))

        #embedding layer
        enc_emb =  Embedding(inputs_length, embedding_dim,trainable=True)(encoder_inputs)

        #encoder lstm 1
        encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        #encoder lstm 2
        encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        #encoder lstm 3
        encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
        encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))

        #embedding layer
        dec_emb_layer = Embedding(outputs_length, embedding_dim,trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)

        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
        decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

        # Attention layer
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concat attention input and decoder LSTM output
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        #dense layer
        decoder_dense =  TimeDistributed(Dense(outputs_length, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Define the model 
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print(self.model.summary())

        #Assign a dictionary that represents each layer
        self.layers = {'encoder_inputs' : encoder_inputs,
                       'enc_emb' : enc_emb,
                       'encoder_lstm1': encoder_lstm1,
                       'encoder_lstm2':encoder_lstm2,
                       'encoder_lstm3':encoder_lstm3,
                       'decoder_inputs' : decoder_inputs,
                       'dec_emb_layer': dec_emb_layer,
                       'decoder_lstm' : decoder_lstm,
                       'attn_layer' : attn_layer,
                       'decoder_dense' : decoder_dense                 
        }

        self.hidden_states = {'encoder lstm 1' : [state_h1, state_c1],
                              'encoder lstm 2' : [state_h2, state_c2],
                              'encoder lstm 3' : [state_h, state_c],
                              'decoder' : [decoder_fwd_state, decoder_back_state],
                              'attn state' : [attn_states]
        }

        self.inputs = {'encoder_inputs' : encoder_inputs,
                        'decoder_inputs' : decoder_inputs
        }

        self.outputs = {'encoder_output1' : encoder_output1,
                        'encoder_output2' : encoder_output2,
                        'encoder_outputs' : encoder_outputs,
                        'decoder_outputs' : decoder_outputs,
                        'attn_out' : attn_out
        }


    def Train(self,x_tr,y_tr,x_val,y_val,epochs = 10,plot_loss_curve = True):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
        #Compile the model
        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        #Train the model
        history=self.model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=epochs,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))
        self.history = history
        #Plot train + val loss
        if plot_loss_curve:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')


    def Inference(self):
        # Encode the input sequence to get the feature vector
        encoder_inputs = self.layers['encoder_inputs']
        #encoder_inputs = Input(shape=(500,))
        state_h,state_c = self.hidden_states['encoder lstm 3'][0], self.hidden_states['encoder lstm 3'][1]

        encoder_model = Model(inputs=encoder_inputs,outputs=[self.outputs['encoder_outputs'], state_h, state_c])
        #print(encoder_model.summery())
        latent_dim = self.latent_dim
        #embedding_dim=100
        max_text_len = 500
        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

        # Get the embeddings of the decoder sequence
        dec_emb2= self.layers['dec_emb_layer'](self.inputs['decoder_inputs']) 
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = self.layers['decoder_lstm'](dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

        #attention inference
        attn_out_inf, attn_states_inf = self.layers['attn_layer']([decoder_hidden_state_input, decoder_outputs2])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = self.layers['decoder_dense'](decoder_inf_concat) 

        # Final decoder model
        decoder_model = Model(
            [self.inputs['decoder_inputs']] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs2] + [state_h2, state_c2])


        return encoder_model,decoder_model



class Preproccesor:
    def __init__(self,data,headers):
        #method gets the data (list of lists) which contains what you want to summarize
        #and the headers (list of lists) which are the summrizations
        self.data = data
        self.headers = headers

    def Run(functions):
        pass


    def ImplementNLTK(self,data,headers):
        nltk.download('stopwords')
        tokenizer = RegexpTokenizer(r'\w+')
        stopword_set = set(stopwords.words('english'))
        pancs_marks = set(string.punctuation)
        new_data = []
        for article in data:
            #remove pancutation marks
            new_str = ''.join(ch for ch in article if ch not in pancs_marks)
            new_str = article.lower()
            dlist = tokenizer.tokenize(new_str)
            #dlist = list(set(dlist).difference(stopword_set))
            new_data.append(dlist)

        new_headers = []
        for header in headers:
            try:
                new_str = ''.join(ch for ch in header if ch not in pancs_marks)
                new_str = header.lower()
                dlist = tokenizer.tokenize(new_str)
                #dlist = list(set(dlist).difference(stopword_set))
                new_headers.append(dlist)
            except:
                print('e')
       
        return new_data,new_headers


    def Lemmatization(self,data,headers,return_as_strings = True):
        nltk.download('wordnet')
        ps = PorterStemmer()
        # define the lemmatiozation object
        lemmatizer = WordNetLemmatizer()

        new_data = []
        for file in data:
            new_file = []
            for word in file:
                stemmed_word = ps.stem(word)
                lemetized_word = lemmatizer.lemmatize(stemmed_word)
                new_file.append(lemetized_word)
            new_data.append(new_file)

        new_headers = []
        for header in headers:
            new_header = []
            for word in header:
                stemmed_word = ps.stem(word)
                lemetized_word = lemmatizer.lemmatize(stemmed_word)
                new_header.append(lemetized_word)
            new_headers.append(new_header)
        
        #If return as strings, return list of strings, else return list of list of words
        if not return_as_strings:
            return new_data,new_headers
        else:
            data = []
            for file in new_data:
                tmp = ''
                for word in file:
                    tmp += word + ' '
                data.append(tmp)
            headers = []
            for header in new_headers:
                tmp = ''
                for word in header:
                    tmp += word + ' '
                headers.append(tmp)

            print('Data : \n',data[:5])
            print('Headers : \n',headers[:5])
            
            return data,headers


    def EliminateLowFrequencies(self,data):
        pass


    def Word2VecSetup(self,data,headers,check_similarities = False):
        data += headers

        print('Initializing the Word2Vec model...')
        model = Word2Vec(
            data,
            size=150,
            window=10,
            min_count=2,
            workers=10,
            iter=10)
        # print('Building vocab...')
        # model.build_vocab(data)

        print('Training the W2V model...')
        st = time.time()

        model.train(data,total_examples=len(data), epochs = 10)
        print('Finished. Time : ', time.time() - st)

        if check_similarities:
            status_str = 'notebook'
            while status_str != 'exit':
                try:
                    print(model.most_similar(status_str)[:5])
                except:
                    print('Word {0} not in vocab.'.format(status_str))
                status_str = input('Enter word')

        return model


    def TokenizeData(self,data,headers,train_percentage = 0.7,val_percentage = 0.15,test_percentage = 0.15):
        x = []
        y = []
        
        
        #Append start and end of line tokens fpr headers
        for i in range(len(headers)):
            headers[i] = 'START ' + headers[i] + 'END'

        
        max_text_len = 500
        max_summary_len = 10

        train_idx = int(len(data) * train_percentage)
        val_idx = int(len(data) * (train_percentage + val_percentage))

        x_tr = data[:train_idx]
        x_val = data[train_idx:val_idx]
        y_tr = headers[:train_idx]
        y_val = headers[train_idx:val_idx]
        x_tokenizer = Tokenizer()
        x_tokenizer.fit_on_texts(list(x_tr))

        thresh = 2

        cnt = 0
        tot_cnt = 0
        freq = 0
        tot_freq = 0

        for key, value in x_tokenizer.word_counts.items():
            tot_cnt = tot_cnt + 1
            tot_freq = tot_freq + value
            if (value < thresh):
                cnt = cnt + 1
                freq = freq + value

        print("% of rare words in vocabulary:", (cnt / tot_cnt) * 100)
        print("Total Coverage of rare words:", (freq / tot_freq) * 100)

        # prepare a tokenizer for reviews on training data
        x_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
        x_tokenizer.fit_on_texts(list(x_tr))

        # convert text sequences into integer sequences
        x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
        x_val_seq = x_tokenizer.texts_to_sequences(x_val)

        # padding zero upto maximum length
        x_tr = pad_sequences(x_tr_seq, maxlen=max_text_len, padding='post')
        x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

        # size of vocabulary ( +1 for padding token)
        x_voc = x_tokenizer.num_words + 1



        ############labels###########
         
        
        
        y_tokenizer = Tokenizer()
        y_tokenizer.fit_on_texts(list(y_tr))
        cnt = 0
        tot_cnt = 0
        freq = 0
        tot_freq = 0

        for key, value in y_tokenizer.word_counts.items():
            tot_cnt = tot_cnt + 1
            tot_freq = tot_freq + value
            if (value < thresh):
                cnt = cnt + 1
                freq = freq + value

        print("% of rare words in vocabulary:", (cnt / tot_cnt) * 100)
        print("Total Coverage of rare words:", (freq / tot_freq) * 100)

        # prepare a tokenizer for reviews on training data
        y_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
        y_tokenizer.fit_on_texts(list(y_tr))

        # convert text sequences into integer sequences
        y_tr_seq = y_tokenizer.texts_to_sequences(y_tr)
        y_val_seq = y_tokenizer.texts_to_sequences(y_val)

        # padding zero upto maximum length
        y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
        y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

        # size of vocabulary
        y_voc = y_tokenizer.num_words + 1

        self.x_voc = x_voc
        self.y_voc = y_voc
        self.x_tokenizer = x_tokenizer
        self.y_tokenizer = y_tokenizer

        return x_tr,x_val,y_tr,y_val,x_voc,y_voc,x_tokenizer,y_tokenizer



    def seq2text(self,input_seq):
        if not hasattr(self,'x_tokenizer'):
            print("Preproccesor's object has no attribute x_tokenizer, call Tokenize data mehotd first.")
            return None
        newString=''
        for i in input_seq:
            if(i!=0):
                newString=newString+self.x_tokenizer.index_word[i]+' '
        return newString
     

    def seq2summary(self,input_seq):
        if not hasattr(self,'y_tokenizer'):
            print("Preproccesor's object has no attribute y_tokenizer, call Tokenize data mehotd first.")
            return None
        newString=''
        for i in input_seq:
            if((i!=0 and i!=self.y_tokenizer.word_index['start']) and i!=self.y_tokenizer.word_index['end']):
                newString=newString+self.y_tokenizer.index_word[i]+' '
        return newString





def PrepareFoodReviewsDataset(path, load_percentage=0.1, using_pandas=True):
    print('Loading food reviews dataset...')
    st = time.time()
    if using_pandas:
        datafile = pd.read_csv(path)
        # Drop missing values
        datafile = datafile.dropna()
        # Split load percentage
        datafile = datafile.iloc[:int(load_percentage * datafile.shape[0])]
        data = datafile['Text'].to_list()
        headers = datafile['Summary'].to_list()
    else:
        reader = csv.reader(open(path, 'r'))
        REVIEW_COL = 9
        SUMMERY_COL = 8
        data = []
        headers = []
        for line in reader:
            data.append(line[REVIEW_COL])
            headers.append(line[SUMMERY_COL])

        data, headers = data[:int(len(data) * load_percentage)], headers[:int(len(headers) * load_percentage)]

    print('Time : ', time.time() - st)
    return data, headers


def GetSummary(input_seq,encoder_model,decoder_model,preproccesor):
    x_tokenizer = preproccesor.x_tokenizer
    y_tokenizer = preproccesor.y_tokenizer
    reverse_target_word_index=y_tokenizer.index_word
    #reverse_source_word_index=x_tokenizer.index_word
    target_word_index=y_tokenizer.word_index
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    max_summary_len = 30
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['start']
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        #Predict the next summary word (as a vector)
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # De-tokenize the vector to get the word
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_summary_len - 1)):
            stop_condition = True
            break

        decoded_sentence += ' '+sampled_token

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states to the next prediction
        e_h, e_c = h, c


    return decoded_sentence


def GenerateSummaries(data,org_summaries,lstm_model,preproccesor):
    #Method gets the trained model and a list of senteneces and outputs thier summaries
    #Split the LSTM model to decoder and encoder
    encoder_model,decoder_model = lstm_model.Inference()
    for i in range(data.shape[0]):
        sentence = data[i]
        decoded_sentence = GetSummary(sentence.reshape(1,500),encoder_model,decoder_model,preproccesor)
        print('Data : \n',preproccesor.seq2text(sentence))
        print('Summery: \n',preproccesor.seq2summary(org_summaries[i]))
        print('Predicted summary : \n', decoded_sentence)



def main():
    print('Processing the data...')
    data,headers = PrepareFoodReviewsDataset(DATAST_FILE_PATH)
    preproccesor = Preproccesor(data,headers)
    print('Cleaning data...')
    cleaned_data,cleaned_headers = preproccesor.ImplementNLTK(data,headers)
    print('lematizing data...')
    lematized_data,lemmatized_headers = preproccesor.Lemmatization(cleaned_data,cleaned_headers)
    print('Removing frequencies...')
    x_tr,x_val,y_tr,y_val,x_voc,y_voc,x_tokenizer,y_tokenizer = preproccesor.TokenizeData(lematized_data,lemmatized_headers)
    print('X train shape : \n',x_tr.shape)
    print('y train shape : \n',y_tr.shape)
    print('X val shape : \n',x_val.shape)
    print('y val shape : \n',y_val.shape)
    #LSTM model
    lstm_model = LSTM_Model(x_voc,y_voc)
    lstm_model.Train(x_tr,y_tr,x_val,y_val)
    GenerateSummaries(x_tr,y_tr,lstm_model,preproccesor)


    print('Done.')


if __name__ == "__main__":
    main()
