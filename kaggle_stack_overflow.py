import datetime
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# replace with correct path
training_path = '/Users/swhitmore/pycon2015-kaggle-tutorial/1/train.csv'
test_path = '/Users/swhitmore/pycon2015-kaggle-tutorial/1/test.csv'
submission_path = '/Users/swhitmore/pycon2015-kaggle-tutorial/submissions/'

def load_feature_matrix(path):
    '''Feature creation:

        We here use a few methods to transform heterogeneous parts of 
        our dataset to numeric features.
    '''
    df = pd.read_csv(path)

    # we can create a feature "length of title"
    # -- will more descriptive titles get more answers?
    df["TitleLength"] = df.Title.apply(len)

    # similar train of thought for including the text length
    df["TextLength"] = df.BodyMarkdown.apply(len)

    # process of creating a feature "Post Time Minus Account Creation Time"
    # -- maybe more seasoned members get better responses
    df["PostCreationDatetime"] = df.PostCreationDate.apply(lambda d: datetime.datetime.strptime(d, "%m/%d/%Y %H:%M:%S"))
    df["PostCreationDateSeconds"] = df.PostCreationDatetime.apply(datetime.date.toordinal)
    df["OwnerCreationDatetime"] = df.OwnerCreationDate.apply(
        lambda d: datetime.datetime.strptime(d, "%m/%d/%Y %H:%M:%S") if '-' not in d else np.nan)
    df["OwnerCreationDateSeconds"] = df.OwnerCreationDatetime.apply(lambda d: datetime.date.toordinal(d) if d else np.nan)
    df["PostTimeMinusAccountCreationTime"] = df["PostCreationDateSeconds"] - df["OwnerCreationDateSeconds"]
    
    # apply a feature to count number of tags on the question
    df["TagNumber"] = df[["Tag1", "Tag2", "Tag3", "Tag4", "Tag5"]].applymap(lambda x: 1 if x is not np.NaN else 0).sum(axis=1)
    return df


def logistic_regression_method():
    '''Logistic regression approach for classification:

        The logistic regression classifier takes in the numeric features we have defined
        in the load_feature_matrix method and fits each feature to a coefficient which is 
        proportional to its importance in determining the OpenStatus in the training set.
        The function is then evaluated and after a cutoff value will be mapped to the output 
        values 0 or 1.
    '''
    train = load_feature_matrix(training_path)
    lr = LogisticRegression()

    # delineating the features we have created or chosen
    columns = ["ReputationAtPostCreation", "TitleLength", "PostTimeMinusAccountCreationTime", "TagNumber", "TextLength"]

    # fitting them to the output vector OpenStatus
    lr.fit(X = np.asarray(train[columns]), y = np.asarray(train.OpenStatus).transpose())
    submit(lr, columns, "logistic_regression_submission.csv")


def naive_bayes_method():
    '''Naive Bayes method for classification:

        This approach comprises of two steps: vectorization and classification.

        First step -- vectorization:

        We use a vectorizer to map the words in the training set messages to a matrix.
        The matrix has rows of words and columns of document ID, and marks wherever a word 
        occurs in a document by changing that matrix element.  We use a TFIDF (text frequency
        inverse document frequency) vectorizer, which in an additional step, enters in weights
        instead of binary values to the matrix which represent the frequency of the word
        in a document divided by the frequency of the word in the entire corpus (so as to mark
        especially rare words as distinctive for a certain document).

        Second step -- classification:
        The classifier begins by assuming that a post has a uniform chance (50/50)
        of being open or closed.  It then updates this probability with the evidence provided
        by the vectorized feature matrix.  For example, if the row for a specific word contains 
        many more posts that are closed than open, the classifier will note that evidence and 
        will output a higher prediction for a future post containing that word to be closed 
        (ignoring other potential mitigating factors for this example).

        The parameters for our pipeline of vectorization and classification are optimized
        using the sklearn GridSearchCV method, which tries all combinations of provided parameters
        and picks the model with the best cross validation score.

    ''' 
    train = load_feature_matrix(training_path)
    test = load_feature_matrix(test_path)
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultinomialNB()),
    ])

    parameters = {
        'vect__max_features': (5000, 10000), # how many features to vectorize
        'vect__ngram_range': ((1,1), (1, 2)),  # unigrams or bigrams
        'clf__alpha': (0.00001, 0.000001), # smoothing parameter for classfier
    }

    # log loss scoring is the same used by the Kaggle competition scoring
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10, scoring='log_loss')

    # we performed Naive Bayes on the text of the question only
    # adding the title might have resulted in additional information!
    grid_search.fit(train['BodyMarkdown'].values, train['OpenStatus'].values)
    new_model = grid_search.best_estimator_
    submit(lr, ["BodyMarkdown"], "naive_bayes_submission.csv")


def submit(model, columns, submission_filename):
    '''Helper method for loading dataframes and submitting probabilities'''
    test = load_feature_matrix(test_path)

    # output probabilistic predictions using the model
    predictions = model.predict_proba(np.asarray(test[columns].values))[:,1]

    # only these two fields are allowed in submissions
    submission = pd.DataFrame({"id": test.PostId, "OpenStatus": predictions})
    submission.to_csv(submission_path + submission_filename, index=False)

