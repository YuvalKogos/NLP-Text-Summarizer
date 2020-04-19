# NLP-Text-Summarizer
Creating a text summarizer using seq2seq-autoencoder-LSTM network architecture.
Developing a RNN (LSTM) network that recieves a text input, procceses it, and generates a summarize output.

The source.py script contains two classes which demonstrates the flow of the process:
1. Preproccesor: preprocess the input text using the following steps:
   - Lower case everything
   - Remove punctuation marks
   - Lemmatizing (convert every word to it's base word, E.g. cars -> cars, working -> work, etc...)
   - Tokenizing
   
2. LSTM_model: The neural net that procces's and generates summarizations:
   The final model built as seq2seq (autoenoder) model, and consists of 2 models - the encoder and the decoder.
   The encoder responsible for analyzing the text and understand the concept of the paragraph, 
   then it passes it's states to the decoder which generates the output summary.


TODO: Try to replace the encoder with Google's pretrained Word2Vec model, and retrain it on my data.

--------------------------------------


The model has been trained on the Amazon's food reviews dataset, and those are some of the results:


Data: great price fast shipping best chips better ingredients less calories snack foods plus taste like real chips \n
Original summary: pop chips are the best \n
Prediction:  amazing recommended chips

Data: dog loves lickety stik bacon flavor since likes much plan getting flavors great liquid treat dog highly recommend lickety stik \n
Original summary: great dog treat \n
Prediction:  dog loves them

Data: good soft drink smooth strawberry cream soda tasty \n
Original summary: good stuff \n
Prediction:  good stuff

Review: drink cups day verona italian french roast coffee wanted try lower acid version brand coffee smells tastes like vinegar totally unpalatable better drinking water acid coffee bothers \n
Original summary: single worst coffee ever \n
Predicted summary:  very bad coffee



*BBC news summarization exmaples will be uploaded soon.
