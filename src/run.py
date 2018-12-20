import preprocessing as preproc
import helpers
import text_representation as text_repr
import numpy as np
import models as models

#choose algorithm to run among {NN, CNN, LSTM}
ml_algorithm = 'LSTM'
# if small you will work with the small datasets
full_dataset = True
# if you want to work with glove-pretrained model by Stanford
glove_pretrained = True

best_model_filepath = "lstm.weights.best.hdf5"
DATA_TWITTER_DATASETS = '../data/twitter-datasets/'

#get processed data, if the processed files already exist no preprocessing will be done
train_data, y, test_data = helpers.get_processed_data(full_dataset=full_dataset)
#define number of words
num_words = text_repr.define_num_words(train_data, full_dataset)

if ml_algorithm in ['NN', 'CNN', 'LSTM']:
    #initialize keras Tokenizer
    tokenizer = text_repr.fit_tokenizer(train_data, num_words)

    word_embeddings = dict()
    if glove_pretrained:
        word_embeddings = text_repr.loadGloveModel(DATA_TWITTER_DATASETS+"glove.twitter.27B.200d.txt")
        embedding_matrix = text_repr.get_embedding_matrix(tokenizer, word_embeddings, num_words)
        print('embedding_matrix.shape', embedding_matrix.shape)
    tweet_max_length = text_repr.define_tweet_max_len(train_data, full_dataset)
    train_data_seq, test_data_seq = text_repr.create_pad_sequences_train_test(tokenizer, train_data, test_data , tweet_max_length)
    print(train_data_seq.shape)
    if(models.best_model_is_provided(best_model_filepath)):
        model = models.return_best_model(best_model_filepath)
    else:
        model = models.fit_NN_model(train_data_seq, y, best_model_filepath)

    helpers.predict_and_save(model, test_data_seq, 'LSTM')