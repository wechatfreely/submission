### Q1
The statement should be run on MySQL.

### Q2
A file named 'log' is read into pandas.

### Q3a
The program first generate a corpus 
for each of the two input csv files: "offsite-tagging-training-set corpus.csv" 
and "offsite-tagging-test-set corpus.csv".
One can then use these two files directly
in the training step.
To generate the corpus files, 
I use CKIP library for the Chinese word segmentation. 
This library specifically caters for Traditional Chinese character.
It is worth exploring the other alternative Corpus Tool, 
e.g. SnowNLP, Jieba, etc.,
which might differ in terms of performance.

For training, I use the bag of words method,
where the occurrence of the words is used as a feature for classification.
I use multinomial Naive Bayes classifier,  
which is an appropriate for handling counting data.

The hyperparameters are tuned by searching over the grid of 
parameters using GridSearchCV, in which the 
accuracy_score is used to evaluate the performance of the training models. 
This measure shows the fraction of correctly classified samples.
To prevent the variance problem, 5 fold cross validation is used. 
The GridSearchCV results are saved in bestparam.pickle, which can be
used for further tests.
After the grid search, the best model is able to predict with 99% accuracy. 

Finally, the best parameters chosen from GridSearchCV
is used run tests on the "offsite-tagging-test-set corpus.csv".