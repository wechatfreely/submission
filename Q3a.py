import pandas as pd
from ckiptagger import WS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pickle

def generate_corpus(raw_data, train=True):
    ws = WS("./data")
    corpus = []
    y_train = []
    i = 1
    for df in pd.read_csv(raw_data, sep=',', header=0, chunksize=1):
        i = i + 1
        print(df)
        docs = df['text'].values[0]
        ws_results = ws([docs])
        all_list = str(' '.join([str(elem) for elem in ws_results[0]]))
        corpus.append(all_list)
        if train == True:
            y = df['tags'].values[0]
            y_train.append(y)
        #if i > 2:
        #    break
    return corpus, y_train


def save_corpus():
    raw_data = "offsite-tagging-training-set (1).csv"
    corpus_train, y_train = generate_corpus(raw_data)
    corpus_df = pd.DataFrame(corpus_train, columns=['text'])
    corpus_y = pd.DataFrame(y_train, columns=['tags'])
    df = pd.concat([corpus_df, corpus_y], axis=1)
    csv_data = df.to_csv('offsite-tagging-training-set corpus.csv', index=False)

    raw_data = "offsite-tagging-test-set (1).csv"
    corpus_test, y_test = generate_corpus(raw_data, train=False)
    df = pd.DataFrame(corpus_test, columns=['text'])
    csv_data = df.to_csv('offsite-tagging-test-set corpus.csv', index=False)

if __name__ == "__main__":
    #This function generate corpus for training. The results are saved as
    #'offsite-tagging-test-set corpus.csv' & 'offsite-tagging-training-set corpus.csv'
    #Uncomment if you want to re-generate the corpus files.
    #save_corpus()

    #read the taining
    raw_data = "offsite-tagging-training-set corpus.csv"
    df =pd.read_csv(raw_data, sep=',', header=0)
    x_train=df['text'].values.tolist()
    y_train =df['tags'].values.tolist()

    #define the training steps
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])
    #define the grid for the hyperparameters
    tuned_parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': [1, 1e-1, 1e-2]
    }

    #train the model and tune the parameters
    clf = GridSearchCV(text_clf, tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(x_train, y_train)

    #shows the training results for different parameters, and save them in csv for reference
    df_clf = pd.DataFrame(clf.cv_results_)
    df_clf.to_csv("best_param.csv")
    print('the best_params : ', clf.best_params_)
    print('the best_score  : ', clf.best_score_)

    # Save model
    pickle.dump(clf, open("bestparam.pickle", "wb"))


    #read the testing set
    raw_data = "offsite-tagging-test-set corpus.csv"
    df = pd.read_csv(raw_data, sep=',', header=0)
    x_test = df['text'].values.tolist()

    #load the model
    clf = pickle.load(open("bestparam.pickle", "rb"))

    #prediction
    y_pred=clf.predict(x_test)
    pd.DataFrame(y_pred).to_csv("prediction.csv")