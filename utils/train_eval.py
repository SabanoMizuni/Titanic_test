from sklearn.model_selection import cross_val_score
from utils.models import create_models
import matplotlib.pyplot as plt

def train_eval(df_X_train, df_y_train, df_X_test, df_y_test):

    """Prep variables"""
    best_score = 0
    best_model = ""
    scores_train = list()
    scores_val = list()
    x_labels = list()

    """Train and Evaluate each model"""
    models = create_models()

    for i, model in enumerate(models):
        score = cross_val_score(model, df_X_train, df_y_train, cv=10)

        model.fit(df_X_train, df_y_train)
        print('---------------------------------')
        print('[Model]', model.__class__.__name__)
        print('Accuracy(train):{:.3f}'.format(score.mean()))
        score_val = model.score(df_X_test, df_y_test)
        print('Accuracy(test):{:.3f}'.format(score_val))

        scores_train.append(score.mean())
        scores_val.append(score_val)
        x_labels.append(model.__class__.__name__)

        """Model Comparison"""
        if model.score(df_X_test, df_y_test) > best_score:
            best_model = model.__class__.__name__
            best_score = model.score(df_X_test, df_y_test)

    """Visualise the result of the models"""
    _visualise_scatter(data=scores_train,
                      x_labels=x_labels,
                      file_name="train_result.png",
                      title="Training Result over CV=5")
    _visualise_scatter(data=scores_val,
                      x_labels=x_labels,
                      file_name="val_result.png",
                      title="Validation Result over CV=5")

    print('---------------------------------')
    print('best_model：', best_model)
    print('best_score：', best_score)

def _visualise_scatter(data, x_labels, file_name, title):
    """ Prep scatter plot """
    plt.scatter(range(len(data)), data)
    plt.title(title)
    plt.grid()
    plt.xticks(range(len(data)), x_labels)
    plt.savefig(file_name)
    #plt.show()
    plt.clf()