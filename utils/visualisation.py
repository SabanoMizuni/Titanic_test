import numpy as np
import matplotlib as plt
import seaborn as sns


def plot_x_and_y(df, train):
    fig = plt.figure(figsize=(14, 9))
    for i in np.arange(12):  # for each column of x
        ax = fig.add_subplot(3, 4, i + 1)  # Add subplot
        sns.countplot(x=df.iloc[:, i], data=train, hue="Survived", palette="Pastel1")
    plt.show()
