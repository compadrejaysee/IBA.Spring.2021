import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chisquare
from sklearn import tree
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import missingno as mano

#F1: loading data in a dataframe (either CSV or Excel - can be generalized for databases)
def import_data(path_to_file):
    file_ext = path_to_file.split(".")[-1].lower()
    if file_ext == "csv":
        return pd.read_csv(path_to_file)
    elif file_ext == "xlsx":
        return pd.read_excel(path_to_file)

#F2: checking shape, column types, and see the first/last 'n' rows using head/tail (where n is one of the arguments of F2)
# df =import_data("E:\Documents\Education\IBA\Courses\Spring 2021\ML 1\Quiz1\House.Price.csv")

def data_summary(df,n):
    print("###### Shape ######")
    print(df.shape)
    print("###### Dimensions ######")
    print(df.ndim)
    print("###### Dtypes ######")
    print(df.dtypes)
    print("###### head ######")
    print(df.head(n))
    print("###### tail ######")
    print(df.tail(n))

# data_summary(df,5)

#F3: remove unnecessary/useless columns (based on results of F2 and your background knowledge and the problem to be solved), e.g., identifiers, multiple primary keys, extra KPI like GMROI in sales which is the same for the whole year etc.

def drop_columns(df, list_of_col):
    return df.drop(list_of_col,axis=1)

#F4: remove rows containing a particular value of a given column, e.g., in smoking_status column, I don't want to consider non-smokers in my ML problem so I remove all these rows.

def drop_rows(df, colname, row_value):
    return df[~df[colname].isin([row_value])]

#F5: determine the missing values in the whole dataset

def missing_values(df):
    print(df.isnull().sum())


# F6: analyze missing values of one or more columns using mano module

def analyze_missing(df):
    mano.matrix(df)
    mano.bar(df)

#F7: cater for missing values (input the column with missing value, and the method through which you want to cater for the missing values)
def median_impute(df, colname):
  median = df[colname].median()
  df[colname].fillna(median, inplace=True)



#F8: Function for numerical data analysis - includes histogram, boxplot, qqplot, describe, and statistical tests for normality
def numerical_analysis(df, list_of_num_cols, hist_bins='auto'):
    print(df[list_of_num_cols].describe())
    print("####### histogram ########")
    for i in list_of_num_cols:
        df[i].hist(grid=True, bins=hist_bins)
        plt.xlabel(i)
        plt.ylabel('Value')
        plt.title(i)
        plt.show()
    print("####### histogram ########")
    for i in list_of_num_cols:
        df.boxplot(column=i, sym='o', return_type='axes')
        plt.show()
    print("####### QQPLOT ########")
    for i in list_of_num_cols:
        sm.qqplot(df[i], line ='45')
        plt.title(i)
        plt.show()
    
    print("########Test for Normality#########")
    for i in list_of_num_cols:
        stat, p_value = chisquare(df[i])
        print( i + " --> p value = " + str(p_value))

    




#F9: Function for categorical data analysis - includes value counts, and bar charts
def categ_analysis(df, list_of_columns):
    for i in list_of_columns:
        print(df[i].value_counts())
        # print(df[i].value_counts().plot.bar())
        print("\n")
        # sns.set(style="darkgrid")
        sns.barplot(df[i].value_counts().index, df[i].value_counts().values)
        plt.title(i)
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel(i, fontsize=12)
        plt.show()
# categ_analysis(df, ["Street", "MSZoning"])

#F10: Function to change the type of any column (input col name and the type you want)
def change_type(df, column, type_col):
    return df[column].astype(type_col)


#F11: Function to change the discretizations of a particular catergorical column, e.g., rename the values, remove space between value names etc.
def change_data(x):
    y = x.strip().split(" ")
    z = "".join(y).tolower()

def change_discretize(df, column_name ):
    return df[column_name].apply(change_data)


#F12: Function for data analysis - extract year, month etc., subtract dates etc. (this function cannot be specified exactly so just add what you believe are the basic things
def basic_data(df, list_of_columns, new_column_name= "", method="add", values="num"):
    if method == "add":
        df[new_column_name] = df[list_of_columns].sum(axis=1)

# basic_data(df, ["MoSold", "SalePrice"] )
# df

#F13: function to make a deep copy of a dataframe
def deep_copy(df):
    return df.copy(deep=True)

#F14: function to encode categorical into numerical (label, ordinal, or onehot)
def encode_categ(df, column, type_enc):
    if type_enc == "label":
        df[column] = df[column].astype('category')
        return df[column].cat.codes
    elif type_enc == "onehot":
        return pd.get_dummies(df,columns=[column])


#F15: function to split dataframe into X (predictors) and y (label), apply standard scaling on X, apply the desired ML algorithm and output the results:
# Â - input dataframe
# Â - input the algo name (e.g., decisiontree)
# Â - input whether this is a classification task or a regression task (then you should select either decisiontreeclassifier or decisiontreeregressor within the function)
# Â - for classification, output confusion matrix, AUC, logloss and classification report
# Â - for regression, output MAE, MSE, R-squared and adjusted R-squaredÂ 
# Â - NB: you can add more metrics if available

def predict_model(df, algo_name, class_or_reg, predictors, labels, n=0, get_score=0):
    X_train, X_test, y_train, y_test = train_test_split(predictors, labels, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    if algo_name == "decisiontree":
        if class_or_reg == "classification":
            
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print("####### confusion matrix ########")
            print(metrics.confusion_matrix(y_test, y_pred))
            print("######## confusion report #########")
            print(materics.classification_report(y_test, y_pred))
            print( "###### log loss ###########")
            print('log loss : ', metrics.log_loss(y_test, y_pred))
            print("######### AUC ########")
            print(roc_auc_score(y_test, y_pred))
        else:
            regressor = tree.DecisionTreeRegressor()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            

    

    elif algo_name == "LinearRegression":
        regr = LinearRegression()
        y_pred = regr.fit(X_train, y_train)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    else:
        models = "KNN"
        classifier = KNeighborsClassifier(n_neighbors=n)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print("####### confusion matrix ########")
        print(metrics.confusion_matrix(y_test, y_pred))
        print("######## confusion report #########")
        print(metrics.classification_report(y_test, y_pred))
        print( "###### log loss ###########")
        print('log loss : ', metrics.log_loss(y_test, y_pred))
        print("######### AUC ########")
        print(metrics.roc_auc_score(y_test, y_pred))
        if get_score == 1:
            return metrics.accuracy_score(y_test, y_pred)

#F16: Function to apply ANOVA and output results
def apply_anova(df, list_of_columns):
    models = ols(list_of_columns[0] + "~+C(" + list_of_columns[1] + ")", data=df).fit()
    tables=anova_lm(models, typ=2)
    return tables

#F17: Function to generate correlation heatmaps
def corr_heatmap(df):
    corrmatrix = df.corr()
    f, axis = plt.subplots(figsize =(15, 10)) 
    sns.heatmap(corrmatrix, ax = axis, linewidths = 0.05)

#F18: Function to generate scatter plot

def gen_scatterploy(df, list_of_columns):
    plt.scatter(df[list_of_columns[0]], df[list_of_columns[1]], c='green')
    plt.show()

