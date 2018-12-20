# ML_Project2 -- Project Text Sentiment Classification

The task of this project is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.

## Project Overview

Through this project we experimented with a lot of possible solutions, concerning tweet and text preprocessing, text representation and classification techniques. Some combinations of the aforementioned algorithms predicted high scores and some others performed poorly. The test phases that we went through gave us a thorough and a clear understanding of each component. So, we managed to achieve **a score of 86.5%** by implementing a Stacked LSTM Neural Network.

You shall easily reproduce our score by running the python script: run.py

## Dependencies
You should run the script with Python 3 and also you need to install the following dependencies, so as to run our script:

 1) download and install Anaconda3-5.3.1 with python 3.6
 1) pip install nltk and then install its packages: wordnet, sentiwordnet,
 punkt, averaged_perceptron_tagger
    ```sh
	$ python
	$ >>> import nltk
	$ >>> nltk.download('package')
	```
2) pip install scikit-learn
3) conda install gensim
4) pip install pandas
5) pip install keras (tested on version 2.2.4)
6) pip install tensorflow (1.12.0)
7) only in case you want to run with the glove_python you shall install 
   pip install glove-python

## Folder Structure (very important part)
* [Train and Test Datasets](https://www.crowdai.org/challenges/epfl-ml-text-classification/dataset_files)
	 please move the three datasets under 'data/twitter-datasets/'

* To avoid the preprocessing phase you shall download the preprocessed tweets of the full dataset from this [link](https://drive.google.com/open?id=1OKkMXY3lN882cOKeEKqHlscyZ09kSMWh)
As soon as you have the files: train_neg_full_processed.txt, train_pos_full_processed.txt and test_data_processed.txt , please move them under 'data/intermediate/'

* To run our pipeline from *scratch* and reproduce the best score you should download the pre-trained word Vectors for twitter from Stanford:
	```sh 
	wget https://nlp.stanford.edu/data/glove.twitter.27B.zip -O data/twitter-datasets/glove.twitter.27B.zip
	unzip data/twitter-datasets/glove.twitter.27B.zip -d data/twitter-datasets/
	```

* Also, to *avoid running our model from scratch* you can download the model from this [link](https://drive.google.com/open?id=1V_xtWUOGT5Qa3uc8sB9hfQDbbczHQ-Fg) and move the file under 'data/intermediate/'

## Hardware Requirements

We implemented the algorithms on our own laptops (Ubuntu) with **16GB** of RAM
But we produced the best score, after renting 1 GPU and we trained our LSTM model for 16 hours (5 epochs).

# Project Structure
------------

    ├── src
    |    ├── run.py                          : Main script to produce our best submission score.
    |
    ├──  |── models.py                       : Script to produce our models, uses LSTM by default.
    |
    ├──  |── cross_validation.py             : Script that performs cross validation.
    |    ├── helpers.py                      : Script that contains all the functions used by our run.py script.
    |
    |    ├── preprocessing.py                : Script that includes all the functions used for preprocessing.
    |
    |    ├── text_representation.py          : Script that includes all our feature representation algorithms.
    ├── README.md                            : The README guideline and explanation for our project.
    |
    ├── data
    │    ├── utils 
    |           ├── words-by-frequency.txt   : lexicon that contains the most used words (important to replace elongated words)
    │   	   ├── slang.csv                 : contains slang words and their replacements
    |		   ├── contractions.csv          : contains contractions and their proper form (replacement)
    ├    ├──intermediate
    │   	├── train_neg_full_processed.txt : processed dataset for negative tweets (after computed or downloaded)
    |		├── train_pos_full_processed.txt : processed dataset for positive tweets (after computed or downloaded)
    |		├── test_data_processed.txt      : processed dataset for test tweets (after computed or downloaded)
    |       ├── lstm.weights.best.txt        : weights for our model (after computed or downloaded)
    ├    ├──twitter-datasets
    |		 ├── glove.twitter.27B.200d.txt  : Pre-trained word vectors of |twitter dataset by Stanford NLP group(after following the |aforementioned commands to download it).
    |        ├── test_data.txt               : Twitter test dataset, containing 10,000 unlabeled tweets.
    |		 ├── train_neg_full.txt          : Twitter training big dataset, containing negative tweets.
    ├        ├── train_pos_full.txt          : Twitter training big dataset, containing positive tweets.
    ├        ├── train_neg.txt               : Twitter training small dataset, containing negative tweets.
    |        ├── train_pos.txt               : Twitter training small dataset, containing positive tweets.
    |   ├── submissions.csv                  : Directory containing submission file generated by run.py.
    |
    ├── report                               : Directory containing our report file.
    |
--------

## Team 

Foivos Anagnou-Misyris : foivos.anagnou-misyris@epfl.ch
Stylianos Agapiou : stylianos.agapiou@epfl.ch
