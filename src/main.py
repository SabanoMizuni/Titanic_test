from utils.prep_dataset import prep_dataset
from utils.train_eval import train_eval

def main():
    df_X_train, df_y_train, df_X_test, df_y_test = prep_dataset()
    train_eval(df_X_train, df_y_train, df_X_test, df_y_test)

if __name__ == '__main__':
    main()
