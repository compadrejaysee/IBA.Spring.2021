import pandas as pd 
import numpy as np
import seaborn as sb
import math
import warnings
import matplotlib.pyplot as plt        
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
from sklearn import preprocessing

#import feature selection modules
from sklearn.feature_selection import mutual_info_classif,RFE,RFECV
from sklearn.feature_selection import mutual_info_regression

#import classification modules
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# import regression modules
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor

#import split methods
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

#import performance scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
#Class Imbalnces
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
#Stats Libraries
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.api as sm


from scipy.stats import shapiro
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# import scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



warnings.filterwarnings("ignore")
sb.set(color_codes=True, font_scale=1.2)

# need to install xgboost first
# pip install xgboost in conda environment
try:
    from xgboost import XGBClassifier
except:
    print("Failed to import xgboost, make sure you have xgboost installed")
    print("Use following command to install it: pip install xgboost")
    XGBClassifier = None

try:
    import lightgbm as lgb
except:
    print("Failed to import lightgbm, make sure that you have lightgbm installed")
    print("Use following command to install it: conda install -c conda-forge lightgbm")
    lgb = None
    
# Load Data
def load_data(file_name):
    def readcsv(file_name):
        return pd.read_csv(file_name)
    def readexcel(file_name):
        return pd.read_excel(file_name)
    def readjson(file_name):
        return pd.read_json(file_name)
    func_map = {
        "csv": readcsv,
        "xls": readexcel,
        "xlsx": readexcel,
        "txt": readcsv,
        "json": readjson
    }
    
    # default reader = readcsv
    reader = func_map.get("csv")
    
    for k,v in func_map.items():
        if file_name.endswith(k):
            reader = v
            break
    return reader(file_name)


# 1) include all possible wrangling functions, including t-tests, chi-squared, anova, correlation heatmaps, normality tests (one of them), #missing value solutions, transformations etc.Ã‚Â




#data cleaning function
def cleaningup(df, to_date=[], to_numeric=[], cols_to_delete=[], fill_na_map={}, cols_to_drop_na_rows=[], cols_to_interpolate=[], corr_threshold=0.7):
    """
    We will perform all the generic cleanup stuff in this function,
    Data specific stuff should be handled by driver program.
    
    Mandatory Parameter:
    df : Dataframe to be cleaned
    
    Optional Parameters:
    to_date:  List of columns to convert to date
    to_numeric:  List of columns to convert to numeric
    cols_to_delete: All the useless columns that we need to delete from our dataset
    fill_na_map:  A dictionary containing map for column and a value to be filled in missing places
                    e.g. {'age': df['age'].median(), 'city': 'Karachi'}
    cols_to_drop_na_rows: List of columns where missing value in not tolerable and we couldn't risk predicting                     value for it, so we drop such rows.
    cols_to_interpolate: List of columns where missing values have to be replaced by forward interpolation
    """
    
    # columns to convert to date format
    def change_type_to_date(df, to_date):
        # Deal with incorrect data in date column
        for i in to_date:
            df[i] = pd.to_datetime(df[i], errors='coerce')
        return df
    
    # columns to convert to numerical format
    def change_type_to_numeric(df, to_numeric):
        # Deal with incorrect data in numeric columns
        for i in to_numeric:
            df[i] = pd.to_numeric(df[i], errors='coerce')
        return df
    
    # columns to delete
    def drop_useless_colums(df, cols_to_delete):
        # Drop useless columns before dealing with missing values
        for i in cols_to_delete:
            df = df.drop(i, axis=1)
        return df
   

    #delete column with are highly correlated
    def drop_highcorrelated_colums(df, corr_threshold):
        col_corr = set() # Set of all the names of deleted columns
        corr_matrix = df.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= corr_threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column
                    col_corr.add(colname)
                    if colname in df.columns:
                        del df[colname] # deleting the column from the dataset
        return df

    #drop all rows which contain more than 40% missing values
    def drop_useless_rows(df):
        min_threshold = math.ceil(len(df.columns)*0.4)
        df = df.dropna(thresh=min_threshold)
        return df
    #######################################
    # delete columns with a single unique value
    def drop_row_uniqueval(df):
        # get number of unique values for each column
        df = pd.DataFrame(df)
        counts = df.nunique()
        #print(counts)
        # record columns to delete
        to_del1 = [i for i,v in enumerate(counts) if v == 1]
        print(to_del1)
        # drop useless columns
        df.drop(to_del1, axis=1, inplace=True)
    
        #df = pd.DataFrame(df1)
        print('First Operation')
        print(df.shape)
        # delete columns where number of unique values is less than 1% of the rows
        # get number of unique values for each column
        counts = df.nunique()
        # record columns to delete
        to_del = [i for i,v in enumerate(counts) if (float(v)/df.shape[0]*100) < 1]
        print(to_del)
        # drop useless columns
        df.drop(to_del, axis=1, inplace=True)
        print('2nd Operation')
        print(df.shape)
    
        # delete rows of duplicate data from the dataset
        df.drop_duplicates(inplace=True)
        print('3rd Operation')
        print(df.shape)
    
        return df
    #########################################
    # drop rows in which columns specified by the driver program has missing values
    def drop_na_rows(df, cols_to_drop_na_rows):
        for i in cols_to_drop_na_rows:
            df = df.drop(df[df[i].isnull()].index)
        return df
    
    # Deal with missing values according to map, e.g., {'age': df['age'].median(), 'city': 'Karachi'}
    def fill_na_vals(df, fill_na_map):
        for col,val in fill_na_map.items():
            df[col].fillna(val, inplace=True)
        return df
    
    # Deal with missing values according to the interpolation
    def fill_na_interpolate(df, cols_to_interpolate):
        for i in cols_to_interpolate:
            df[i] = df[i].interpolate(method ='linear', limit_direction ='forward')
        return df
    
    try:
        df = change_type_to_date(df, to_date)
        df = change_type_to_numeric(df, to_numeric)
        df = drop_useless_colums(df, cols_to_delete)
        df = drop_useless_rows(df)
        df = drop_na_rows(df, cols_to_drop_na_rows)
        df = fill_na_vals(df, fill_na_map)
        df = fill_na_interpolate(df, cols_to_interpolate)
        df = drop_highcorrelated_colums(df, corr_threshold)
        
        
        print("df is all cleaned up..")
        return df
    except Exception as e:
        print("Failed to perform cleanup, exception=%s" % str(e))
    finally:
        return df


#basic analysis
def basicanalysis(df):
    print("Shape is:\n", df.shape)
    print("\n Columns are:\n", df.columns)
    print("\n Types are:\n", df.dtypes)
    print("\n Statistical Analysis of Numerical Columns:\n", df.describe())

#string column analysis analysis
def stringcolanalysis(df):
    stringcols = df.select_dtypes(exclude=[np.number, "datetime64"])
    fig = plt.figure(figsize = (8,10))
    for i,col in enumerate(stringcols):
        fig.add_subplot(4,2,i+1)
        fig.savefig('Categorical.png')
        df[col].value_counts().plot(kind = 'bar', color='black' ,fontsize=10)
        plt.tight_layout()
        plt.title(col)

#numerical analysis
def numcolanalysis(df):
    numcols = df.select_dtypes(include=np.number)
    
    # Box plot for numerical columns
    for col in numcols:
        fig = plt.figure(figsize = (5,5))
        sb.boxplot(df[col], color='grey', linewidth=1)
        plt.tight_layout()
        plt.title(col)
        plt.savefig("Numerical.png")
    
    # Lets also plot histograms for these numerical columns
    df.hist(column=list(numcols.columns),bins=25, grid=False, figsize=(15,12),
                 color='#86bf91', zorder=2, rwidth=0.9)

# Perform correlation analysis over numerical columns
def correlation_anlysis(df):
    # NOTE: If label column is non-numeric, 'encode' it before calling this function 
    numcols = df.select_dtypes(include=np.number)
    corr = numcols.corr()
    ax = sb.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sb.diverging_palette(20, 220, n=200),
    square=True
    )
    
    ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

    
    
    
def correlation(df):
    numcols = df.select_dtypes(include=np.number)
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
    
    
# T-Test
# Conducted t-test for between all numeric columns in dataset

def t_test(df):
    print ("t-test for equality of mean between all numric columns")
    types_map = df.dtypes.to_dict()
    num_columns = []
    for k,v in types_map.items():
        if np.issubdtype(np.int64, v) or np.issubdtype(np.float64, v):
            num_columns.append(k)

    print(num_columns)

    for i in range(len(num_columns)-1):
        for j in range(i+1,len(num_columns)):
            col1 = num_columns[i]
            col2 = num_columns[j]
            t_val, p_val = stats.ttest_ind(df[col1], df[col2])
            print("(%s,%s) => t-value=%s, p-value=%s" % (num_columns[i], num_columns[j], str(t_val), str(p_val)))
     
   
   
####################3
# Normality Test for all Numeric Columns
def Normality_test(df):
    print ("Normality Test for all numric columns")
    types_map = df.dtypes.to_dict()
    num_columns = []
    for k,v in types_map.items():
        if np.issubdtype(np.int64, v) or np.issubdtype(np.float64, v):
            num_columns.append(k)

    print(num_columns)

    for i in range(len(num_columns)):
        col1 = num_columns[i]
        stat, p_val = shapiro(df[col1])
        print('Normaility Test for Column:', col1)
        print('Statistics=%.3f, p_value=%.3f' % (stat, p_val))
        #print("(%s,%s) => Statistics=%s, p-value=%s" % (str(stat), str(p_val)))
        alpha = 0.05
        if p_val > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
        
#######################3
#ANOVA analysis          
def ANOVA_analysis(df):
    categorical = df.select_dtypes(exclude=[np.number, "datetime64"])
    numcols = df.select_dtypes(include=np.number)
    #print('Categorical Colums:', categorical)
    #print('Numerical Colums:', numcols)
    # ANOVA for categorical columns
    for i in categorical:
        for j in numcols:
            col1 = categorical[i]
            col2 = numcols[j]
            model = ols('col2 ~ C(Q("%s"))' % i, data=df[col2]).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print ("\nAnova => - %s" % i)
            display(anova_table)    
    
#######################################
#Chi Square Test for indepence
def chisquare_test(df):
    print ("Chisquare-test for Independence between all numric columns")
    types_map = df.dtypes.to_dict()
    num_columns = []
    for k,v in types_map.items():
        if np.issubdtype(np.int64, v) or np.issubdtype(np.float64, v):
            num_columns.append(k)

    #print(num_columns)

    for i in range(len(num_columns)-1):
        for j in range(i+1,len(num_columns)):
            col1 = num_columns[i]
            col2 = num_columns[j]
            crosstab = pd.crosstab(df[col1], df[col2])

            stat, p, dof, expected = chi2_contingency(crosstab)
            #t_val, p_val = stats.ttest_ind(df[col1], df[col2])
            print("(%s,%s) => chisqr-value=%s, p-value=%s" % (num_columns[i], num_columns[j], str(stat), str(p)))
            alpha = 0.05
            if p <= alpha:
                print('Dependent (reject H0)')
            else:
                print('Independent (H0 holds true)')
     
    
############################################
# Apply label encoding on specified columns
def apply_label_encoding(df, cols=[]):
    le = preprocessing.LabelEncoder()
    for i in cols:
        le.fit(df[i])
        df[i] = le.transform(df[i])
    return df


# One-Hot/dummy encoding on specified columns
def onehotencoding(df):
    df = pd.get_dummies(df)
    return df

# One Hot encoding with Pandas categorical dtype
def onehotencoding_v2(df, cols=[]):
    for col in cols:
        df[col] = pd.Categorical(df[col])
        dfDummies = pd.get_dummies(df[col], prefix = col)
        df = pd.concat([df, dfDummies], axis=1)
        df = df.drop(col, axis=1)
    return df

#2) include all classification and regression algorithms
# Classification Algorithms

def LogReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = LogisticRegression()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def KNN(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = KNeighborsClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def GadientBoosting(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GradientBoostingClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def AdaBoost(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def SVM(trainX, testX, trainY, testY, svmtype="SVC", verbose=True, clf=None):
    # for one vs all
    if not clf:
        if svmtype == "Linear":
            clf = svm.LinearSVC()
        else:
            clf = svm.SVC()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def DecisionTree(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = DecisionTreeClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def RandomForest(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = RandomForestClassifier()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def NaiveBayes(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GaussianNB()
    clf.fit(trainX , trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def MultiLayerPerceptron(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = MLPClassifier(hidden_layer_sizes=5)
    clf.fit(trainX,trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def XgBoost(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = XGBClassifier(random_state=1,learning_rate=0.01)
    clf.fit(trainX,trainY)
    return validationmetrics(clf,testX,testY,verbose=verbose)

def LightGbm(trainX, testX, trainY, testY, verbose=True, clf=None):
    d_train = lgb.Dataset(trainX, label=trainY)
    params = {}
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
    clf = lgb.train(params, d_train, 100)
    return validationmetrics(clf,testX,testY,verbose=verbose)


# Regression Algorithms
def LinearReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = LinearRegression()
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def RidgeReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = Ridge(alpha=0.01)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def LassoReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = Lasso(alpha=0.01)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

#ElasticNet Regression
def ElasticNet(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = ElasticNet(alpha=0.01)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def RandomForestReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = RandomForestRegressor(n_estimators=100)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def PolynomialReg(trainX, testX, trainY, testY, degree=3, verbose=True, clf=None):
    poly = PolynomialFeatures(degree = degree)
    X_poly = poly.fit_transform(trainX)
    poly.fit(X_poly, trainY)
    if not clf:
        clf = LinearRegression() 
    clf.fit(X_poly, trainY)
    return validationmetrics_reg(clf, poly.fit_transform(testX), testY, verbose=verbose)

def SupportVectorRegression(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = SVR(kernel="rbf")
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def DecisionTreeReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = DecisionTreeRegressor()
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def GradientBoostingReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = GradientBoostingRegressor()
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def AdaBooostReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf = AdaBoostRegressor(random_state=0, n_estimators=100)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

def VotingReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100)
    sv = SVR(kernel="rbf")
    dt = DecisionTreeRegressor()
    gb = GradientBoostingRegressor()
    ab = AdaBoostRegressor(random_state=0, n_estimators=100)
    if not clf:
        clf = VotingRegressor([('rf', rf), ('dt', dt), ('gb', gb), ('ab', ab)])
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)


#3) FS: feature selection algorithms (I included RFFS, RFE, MI and PCA but you can add only RFFS as for me, it works good).Ã‚Â
# Helper function to get fetaure importance metrics via Random Forest Feature Selection (RFFS)
def RFfeatureimportance(df, trainX, testX, trainY, testY, trees=35, random=None, regression=False):
    if regression:
        clf  = RandomForestRegressor(n_estimators=trees, random_state=random)
    else:
        clf  = RandomForestClassifier(n_estimators=trees, random_state=random)
    clf.fit(trainX,trainY)
    #validationmetrics(clf,testX,testY)
    res = pd.Series(clf.feature_importances_, index=df.columns.values).sort_values(ascending=False)*100
    print(res)
    return res


#4) CV: cross validation approaches (include stratified K-fold definitely)

# #### For Cross Validation, lets create generator functions for different cross validation techniques, this will help us run an iterator over all folds
def cross_valid_kfold(X, y, split=10, random=None, shuffle=False):
    """
    Generator function for KFold cross validation
    
    """
    kf = KFold(n_splits=split, random_state=random, shuffle=shuffle)
    for train_index, test_index in kf.split(X):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY
    
def cross_valid_repeated_kf(X, y, split=10, random=None, repeat=10):
    """
    Generator function for Repeated KFold cross validation
    
    """
    kf = RepeatedKFold(n_splits=split, random_state=random, n_repeats=repeat)
    for train_index, test_index in kf.split(X):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY
        

def cross_valid_stratified_kf(X, y, split=10, random=None, shuffle=False):
    """
    Generator function for Stratified KFold cross validation
    
    """
    skf = StratifiedKFold(n_splits=split, random_state=random, shuffle=shuffle)
    for train_index, test_index in skf.split(X, y):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY


def cross_valid_strat_shuffle_kf(X, y, split=10, random=None):
    """
    Generator function for StratifiedShuffle cross validation
    
    """
    sss = StratifiedShuffleSplit(n_splits=split, random_state=random)
    for train_index, test_index in sss.split(X, y):
        trainX, testX = X.iloc[train_index], X.iloc[test_index] 
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        yield trainX,trainY,testX,testY

## 5) REG: regularization (lasso, ridge regression) methods 

#Ridge Regression
def RidgeReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = Ridge(alpha=0.01)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)

#Lasso Regression
def LassoReg(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = Lasso(alpha=0.01)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)
#ElasticNet Regression
def ElasticNet(trainX, testX, trainY, testY, verbose=True, clf=None):
    if not clf:
        clf  = ElasticNet(alpha=0.01)
    clf.fit(trainX , trainY)
    return validationmetrics_reg(clf, testX, testY, verbose=verbose)


#6) Include methods for addressing class imbalance¶


# 1. Random under-sampling with imblearn
def Random_UnderSampling(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
    X_rus, y_rus = rus.fit_resample(X, y)
    X=X_rus
    y=y_rus
    z1=pd.DataFrame(X)
    z2=pd.DataFrame(y)
    z3= pd.concat([z1,z2], axis=1)

    return z3 


# 2. Random over-sampling with imblearn
def Random_OverSampling(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    ros = RandomOverSampler(random_state=42)

    # fit predictor and target variable
    X_ros, y_ros = ros.fit_resample(X, y)
    X=X_ros
    y=y_ros
    z1=pd.DataFrame(X)
    z2=pd.DataFrame(y)
    z3= pd.concat([z1,z2], axis=1)
    return z3


# 3. Synthetic Minority Oversampling Technique (SMOTE)

def SMOT_OverSampling(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    smote = SMOTE()
    # fit predictor and target variable
    X_smote, y_smote = smote.fit_resample(X, y)
    X=X_smote
    y=y_smote
    z1=pd.DataFrame(X)
    z2=pd.DataFrame(y)
    z3= pd.concat([z1,z2], axis=1)
    return z3

# 4. NearMiss

def NearMiss_Resamplin(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    nm = NearMiss()
    X_nm, y_nm = nm.fit_resample(X, y)
    X=X_nm
    y=y_nm
    z1=pd.DataFrame(X)
    z2=pd.DataFrame(y)
    z3= pd.concat([z1,z2], axis=1)
    return z3


# 7) Include any other helper/logistic function you want to add
# Helper Function 1: Helpinf Function for Validation Matrix
# Validation metrics for classification
def validationmetrics(model, testX, testY, verbose=True):   
    predictions = model.predict(testX)
    
    if model.__class__.__module__.startswith('lightgbm'):
        for i in range(0, predictions.shape[0]):
            predictions[i]= 1 if predictions[i] >= 0.5 else 0
    
    #Accuracy
    accuracy = accuracy_score(testY, predictions)*100
    
    #Precision
    precision = precision_score(testY, predictions, pos_label=1, labels=[0,1])*100
    
    #Recall
    recall = recall_score(testY, predictions,pos_label=1,labels=[0,1])*100
    
    #get FPR (specificity) and TPR (sensitivity)
    fpr , tpr, _ = roc_curve(testY, predictions)
    
    #AUC
    auc_val = auc(fpr, tpr)
    
    #F-Score
    f_score = f1_score(testY, predictions)
    
   
    if verbose:
        print("Prediction Vector: \n", predictions)
        print("\n Accuracy: \n", accuracy)
        print("\n Precision of event Happening: \n", precision)
        print("\n Recall of event Happening: \n", recall)
        print("\n AUC: \n",auc_val)
        print("\n F-Score:\n", f_score)
        #confusion Matrix
        print("\n Confusion Matrix: \n", confusion_matrix(testY, predictions,labels=[0,1]))
        #ROC plot
        
    res_map = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc_val": auc_val,
                "f_score": f_score,
                "model_obj": model
                
              }
    return res_map


#Validation metrics for Regression algorithms
def validationmetrics_reg(model,testX,testY, verbose=True):
    predictions = model.predict(testX)
    
    # R-squared
    r2 = r2_score(testY,predictions)
    
    # Adjusted R-squared
    r2_adjusted = 1-(1-r2)*(testX.shape[0]-1)/(testX.shape[0]-testX.shape[1]-1)
    
    # MSE
    mse = mean_squared_error(testY,predictions)
    
    #RMSE
    rmse = math.sqrt(mse)
    
    if verbose:
        print("R-Squared Value: ", r2)
        print("Adjusted R-Squared: ", r2_adjusted)
        print("RMSE: ", rmse)
    
    res_map = {
                "r2": r2,
                "r2_adjusted": r2_adjusted,
                "rmse": rmse,
                "model_obj": model
              }
    return res_map


##Helper Function2: to provide list of supported algorithms for Classification and Regression
# Helper function to provide list of supported algorithms for Classification
def get_supported_algorithms():
    covered_algorithms = [LogReg, KNN, GadientBoosting, AdaBoost,
                          SVM, DecisionTree, RandomForest, NaiveBayes,
                          MultiLayerPerceptron,LightGbm]
    #,LightGbm,XgBoost
    if XGBClassifier:
        covered_algorithms.append(XgBoost)
    if lgb:
        covered_algorithms.append(LightGbm)
    return covered_algorithms

# Helper function to provide list of supported algorithms for Regression
def get_supported_algorithms_reg():
    covered_algorithms = [LinearReg, RidgeReg, LassoReg, ElasticNet, RandomForestReg, PolynomialReg, SupportVectorRegression,
                          DecisionTreeReg, GradientBoostingReg, AdaBooostReg, VotingReg]
    return covered_algorithms


#Helper Function3: For Training Test Split



#Train Test Split: splitting manually
def traintestsplit(df,split,random=None, label_col=''):
    #make a copy of the label column and store in y
    y = df[label_col].copy()
    
    #now delete the original
    X = df.drop(label_col,axis=1)
    
    #manual split
    trainX, testX, trainY, testY= train_test_split(X, y, test_size=split, random_state=random)
    return X, trainX, testX, trainY, testY

#helper function which only splits into X and y
def XYsplit(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    return X,y


####################################
#Helper Function for Tranformation

#MinMax Transformation
def MinMax_Transformation(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    scaler = MinMaxScaler()
    scaled_features = MinMaxScaler().fit_transform(X.values)
    df1= pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
    z1=pd.DataFrame(df1)
    z2=pd.DataFrame(y)
    df= pd.concat([z1,z2], axis=1)
    return df 
#Standard Transformation
def Standard_Transformation(df, label_col):
    y = df[label_col].copy()
    X = df.drop(label_col,axis=1)
    scaled_features = StandardScaler().fit_transform(X.values)
    df1= pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
    z1=pd.DataFrame(df1)
    z2=pd.DataFrame(y)
    df= pd.concat([z1,z2], axis=1)
    return df 


#####################################







#8) Include function for executing ML (classification) without FS, REG and CV. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC #curve

# Helper function to run all algorithms provided in algo_list over given dataframe, without cross validation
# By default it will run all supported algorithms 
def run_algorithms(df, label_col, algo_list=get_supported_algorithms(), feature_list=[]):
    """
    Run Algorithms without FS, REG and CV
    
    """
    # Lets make a copy of dataframe and work on that to be on safe side 
    _df = df.copy()
    
    if feature_list:
        impftrs = feature_list
        impftrs.append(label_col)
        _df = _df[impftrs]
    
    _df, trainX, testX, trainY, testY = traintestsplit(_df, 0.2, 91, label_col=label_col)
    algo_model_map = {}
    for algo in algo_list:
        print("============ " + algo.__name__ + " ===========")
        res = algo(trainX, testX, trainY, testY)
        algo_model_map[algo.__name__] = res.get("model_obj", None)
        print ("============================== \n")
    
    return algo_model_map

#9) Include function for executing ML (classification) with FS and without REG and CV. Output: Accuracy, precision (+ve), recall (+ve), AUC, #ROC curve

# Helper function to select important features via RFFS, run supported ML algorithms over dataset with manual split and measure accuracy without Cross Validation - select features with importance >=threshold
def MachineLearningwithRFFS(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    """
    Run Algorithms with Features Selection 
    
    """
    df_cpy = df.copy()
    df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    res = RFfeatureimportance(df_cpy, trainX, testX, trainY, testY, trees=10, regression=regression)
    
    impftrs = list(res[res > threshold].keys())
    #impftrs.append(label_col)
    
    print ("Selected Features =" + str(impftrs))
    print(df.shape)
    results = run_algorithms(df, label_col, algo_list=algo_list, feature_list=impftrs)
    return {"selected_features": impftrs, "results": results}



# 10) Include function for executing ML (classification) with REG and without FS and CV. Output: Accuracy, precision (+ve), recall (+ve), AUC, ROC curve

#Not Included

#11) Include function for executing ML (classification) with CV and without FS and REG. Output: Accuracy, precision (+ve), recall (+ve), #AUC, ROC curve
# With stratified kfold validation support
def run_algorithms_cv(df, label_col, algo_list=get_supported_algorithms(), feature_list=[], cross_valid_method=cross_valid_stratified_kf):
    """
    Run Algorithms with stratified kfold cross validation
    
    """
    _df = df.copy()
    X,y = XYsplit(_df, label_col)
    
    # Select features if specified by driver program
    if feature_list:
        X = X[feature_list]
    
    result = {}
    algo_model_map = {}
    for algo in algo_list:
        clf = None
        result[algo.__name__] = dict()
        for trainX,trainY,testX,testY  in cross_valid_method(X, y, split=10):
            res_algo = algo(trainX, testX, trainY, testY, verbose=False, clf=clf)
            # Get trained model so we could use it again in the next iteration
            clf = res_algo.get("model_obj", None)
            
            for k,v in res_algo.items():
                if k == "model_obj":
                    continue
                if k not in result[algo.__name__].keys():
                    result[algo.__name__][k] = list()
                result[algo.__name__][k].append(v)
                
        algo_model_map[algo.__name__] = clf
            
    score_map = dict()
    # let take average scores for all folds now
    for algo, metrics in result.items():
        print("============ " + algo + " ===========")
        score_map[algo] = dict()
        for metric_name, score_lst in metrics.items():
            score_map[algo][metric_name] = np.mean(score_lst)
        print(score_map[algo])
        print ("============================== \n")
        score_map[algo]["model_obj"] = algo_model_map[algo]
    
    return score_map


#12) Include function for executing ML (classification) with CV and with FS and with REG. Output: Accuracy, precision (+ve), recall (+ve), #AUC, ROC curve

# Helper function to select important features via RFFS, run supported ML algorithms over dataset with cross validation and measure accuracy --- select features with importance >=threshold
def MachineLearningwithRFFS_CV(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    """
    Run Algorithms with stratified kfold CV, FS 
    
    """
    df_cpy = df.copy()
    df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    res = RFfeatureimportance(df_cpy, trainX, testX, trainY, testY,
                              trees=10, regression=regression)

    impftrs = list(res[res > threshold].keys())
    
    print ("Selected Features =" + str(impftrs))
    print(df.shape)
    if regression:
        cross_valid_method = cross_valid_kfold
    else:
        cross_valid_method = cross_valid_stratified_kf
    results = run_algorithms_cv(df, label_col, algo_list=algo_list, feature_list=impftrs, cross_valid_method=cross_valid_method)
    return {"selected_features": impftrs, "results": results}
    

    
## 8-12 above can be merged as a single function as well





def all_CImbalance(df, label_col, algo_list=get_supported_algorithms(), threshold=5, cross_valid_method=cross_valid_stratified_kf, feature_list=[]):
    print("Results without FS, REG and CV")
    R1=run_algorithms(df, label_col, algo_list=get_supported_algorithms(), feature_list=[])
    print("Results with Features Selection")
    R2=MachineLearningwithRFFS(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False)
    print("Results with stratified kfold cross validation")
    R3=run_algorithms_cv(df, label_col, algo_list=get_supported_algorithms(), feature_list=[], cross_valid_method=cross_valid_stratified_kf)
    print("Results with stratified kfold CV, FS")
    R4=MachineLearningwithRFFS_CV(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False)
    return {R1, R2, R3, R4}
