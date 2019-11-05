from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def create_models():
    """ Instantiate models """
    models = []

    models.append(LogisticRegression())
    models.append(DecisionTreeClassifier(criterion='entropy',
                                         max_depth=5,
                                         random_state=0))
    models.append(KNeighborsClassifier(n_neighbors=4))
    models.append(LinearSVC())

    return models