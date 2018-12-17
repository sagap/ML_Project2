import preprocessing as preproc
import helpers
from tqdm import tqdm
import time


start_time = time.time()

full_dataset = False

train_data, y, test_data = helpers.get_processed_data(full_dataset=full_dataset)

model, X_test = helpers.transform_and_fit(train_data, y, test_data, text_representation='word2vec',
                                        ml_algorithm='LR', cross_val=True, predefined=False)
helpers.predict_and_save(model, X_test)

elapsed_time = divmod(round((time.time() - start_time)), 60)
print('------\nElapsed time: {m} min {s} sec\n'.format(m=elapsed_time[0], s=elapsed_time[1]))
