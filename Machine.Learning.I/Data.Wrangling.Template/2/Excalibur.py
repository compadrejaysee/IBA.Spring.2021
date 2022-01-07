#!/usr/bin/env python
# coding: utf-8

#import basic modules
import pandas as pd 
import numpy as np
import seaborn as sb
import math
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  
import seaborn as sns

#mano module
import missingno as mano

#imputer
from sklearn.impute import SimpleImputer

#ggplot
from plotnine import ggplot, aes, geom_point

#Anova, stats modules
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

#chi-square
from scipy.stats import chi2_contingency
from scipy.stats import chi2


from sklearn import preprocessing

#import feature selection modules
from sklearn.feature_selection import mutual_info_classif,RFE,RFECV
from sklearn.feature_selection import mutual_info_regression

# import scaling
from sklearn.preprocessing import StandardScaler

#Encoder module
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

#import split methods
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

#Scaler module
from sklearn.preprocessing import MinMaxScaler

#logistic regression
from sklearn.linear_model import LogisticRegression

#SVM
from sklearn.svm import SVC

#Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier

#linear regression
from sklearn.linear_model import LinearRegression

#KNN classifier
from sklearn.neighbors import KNeighborsClassifier

#Random Forest
from sklearn.ensemble import RandomForestClassifier

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

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

#import performance scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error, r2_score

#regularization module
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.feature_selection import SelectFromModel

# import library class balancing
import imblearn
from imblearn.under_sampling import RandomUnderSampler


warnings.filterwarnings("ignore")
sb.set(color_codes=True, font_scale=1.2)


# In[ ]:


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


# # Function 1: For loading data

# In[1]:


def load_file(filepath):
    if filepath.lower().endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    return df


# # Function 2: for checking basic information of dataset

# In[208]:


def check_dataset(dataset, num_of_rows):
    head = dataset.head(num_of_rows)
    tail = dataset.tail(num_of_rows)
    shape = dataset.shape
    return head, tail, shape


# # Function 3: remove unnecessary/useless columns

# In[212]:


def remove_col(dataset, col):
    dataset= dataset.drop(columns=[col])
    return dataset


# # Function 4: remove rows containing a particular value of a given column

# In[205]:


def remove_rows(value, dataset, col):
    dataset.drop(dataset.index[dataset[col] == value], inplace = True)
    return dataset


# # Function 5: Finding missing values in dataset

# In[74]:


def num_missing(dataset):
    return sum(dataset.isnull())


# # Function 6: Analyzing missing values using mano module

# In[83]:



def mano_analyze(dataset, module):
    module(dataset)
    return 


# # Function 7: Handling missing values
def missing_val(dataset, column , method):
    if method == 'drop':
        for i in column:
            dataset[i] = dataset[i].dropna(axis=0, subset=[column])
    elif method == 'imputation':
        #create a separate data frame for mean imputation
        mean_impute = SimpleImputer(strategy='mean')
        #take only columns where mean imputation matters, i.e., numerical columns
        dataset[[column]] = mean_impute.fit_transform(dataset[[column]])
    elif method == 'interpolate':
        for i in column:
            dataset[i] = dataset[i].interpolate(method ='linear', limit_direction ='forward')
    return dataset



# # Function 8: numerical data analysis - includes histogram, boxplot, qqplot, describe, and statistical tests for normality

# In[47]:


def num_data_analysis(dataset):
    #Box plot of numerical vaiables
    bit = dict(markerfacecolor='b', marker='p')
    boxplot = dataset.boxplot (figsize=(18,10), grid=False, flierprops=bit) 
    #getting numerical columns
    newdf = dataset.select_dtypes(include=np.number)
    for i in newdf.columns:
        print([i])
        #finding normal test of numerical columns
        a,b = stats.normaltest(newdf[i])
        print(a, b)
        if b < 0.5:
            print("The null hypothesis can be rejected")
        else:
            print("The null hypothesis cannot be rejected")
            
     #histogram of numerical vaiables
    dataset.hist(layout=(5,4), color='blue', figsize=(15,12), grid=False)
           
    return dataset.describe


# In[4]:


def ggplot_funct(var1, var2):
    return ggplot(salesdf, aes(x=var1, y=var2)) + geom_point()


# # Function 9: Function for categorical data analysis - includes value counts, and bar charts
def cate_data_analysis(dataset, num_col,cate_col):
    #ploting data
    sn.barplot(x=cate_col ,y=num_col ,data=dataset)
    #count_value
    print(dataset[cate_col].value_counts())
    return plt.show()





def count_all_cate_values(dataset):
    newdf = dataset.select_dtypes(exclude=np.number)
    for i in newdf.columns:
        print(newdf[i].value_counts(10))


# # Function 10: Function to change the type of any column (input col name and the type you want)        

def change_type(cols, typed, dataset):
    if typed == 'category':
         for i in cols:
            dataset[i] = dataset[i].astype('category')
    elif typed == 'numeric':
          for i in cols:
                dataset[i] = pd.to_numeric(dataset[i], errors='coerce')
    elif typed == 'date':
        for i in cols:
            dataset[i] = pd.to_datetime(dataset[i], errors='coerce')
    return dataset
            

# # Function 11: Function to change the discretizations of a particular catergorical column
def rename_categ(dataset, col_name, old_value=0, new_value=0, removespace=0):
    dataset[col_name] = dataset[col_name].replace([old_value],new_value)
    if removespace == 1:
        dataset.replace(" ", "")


# # Function 12: Function for data analysis - extract year, month etc
def date_analysis(dataset, col):
    dataset[col] = pd.to_datetime(dataset[col], errors='coerce')
    salesdf['Quarter'] = dataset[col].apply(lambda x: x.quarter)
    salesdf['Year'] = dataset[col].apply(lambda x: x.year)
    salesdf['Month'] = dataset[col].apply(lambda x: x.month)
    salesdf['Week'] = dataset[col].apply(lambda x: x.week)
    salesdf['Day'] = dataset[col].apply(lambda x: x.day)
    return dataset.tail(5)


# In[160]:


#date_col can be change with Year, Month, Quarter, Day
def date_plot(date_col, col, dataset):
    change_type('SALE PRICE', 'numeric', salesdf)
    agg_qtr = dataset.pivot_table(columns=[date_col], values=col, aggfunc='sum').round(1)
    print(agg_qtr)
    return agg_qtr.plot.bar()


# # Function 13: function to make a deep copy of a dataframe

# In[164]:


def deep_copy(dataset):
    cpy_dataset = dataset.copy(deep=True)
    return cpy_dataset




# One-Hot/dummy encoding on specified columns
def onehotencoding(dataset):
    dataset = pd.get_dummies(dataset)
    return dataset

# # Function 14: function to encode categorical into numerical (label, ordinal, or onehot)
#this function can transform labels, ordered and unordered columns
def cat_data_encoder(dataset, col, val, categories=0):
    if val == 'unordered':
        value = OneHotEncoder(sparse=False)
        dataset[col] = value.fit_transform(dataset[[col]])
    elif val == 'ordered':
        value= OrdinalEncoder(categories=[categories])
        dataset[col] = value.fit_transform(dataset[[col]])


# # Function 15: Function to split dataframe into X (predictors) and y (label)

# The below function can perform classification algorithm (SVM, LogisticRegression, AdaBoostClassifier) and regression (LinearRegression)

# In[8]:


def scaling(X):
      # Scaling all the variables to a range of 0 to 1
        features = X.columns.values
        scaler = MinMaxScaler(feature_range = (0,1))
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = features
        
def results(y_test, preds):
        print('Results')
        print("accuracy: ", metrics.accuracy_score(y_test, preds))
        print("precision: ", metrics.precision_score(y_test, preds)) 
        print("recall: ", metrics.recall_score(y_test, preds))
        print("f1: ", metrics.f1_score(y_test, preds))
        print("area under curve (auc): ", metrics.roc_auc_score(y_test, preds))
        print("Confusion Matrix): ",confusion_matrix(y_test,preds)) 
        
        
        print("AUC Plot")
        test_fpr, test_tpr, te_thresholds = roc_curve(y_test, preds)
        plt.plot(test_fpr, test_tpr)
        plt.plot([0,1],[0,1],'g--')
        plt.legend()
        plt.xlabel("True Positive Rate")
        plt.ylabel("False Positive Rate")
        plt.title("AUC(ROC curve)")
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        plt.show()

def print_evaluate(y_test, preds):  
    mae = metrics.mean_absolute_error(y_test, preds)
    mse = metrics.mean_squared_error(y_test, preds)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, preds))
    r2_square = metrics.r2_score(y_test, preds)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
        

def ML_algo(dataset, label_col, algo_type, algo):
    #reading file in to a dataframe

    if algo_type == 'classification':
        #splitting data in labels and predictors
        y = dataset[label_col].values
        X = dataset.drop(columns = [label_col])
        
        # Scaling all the variables to a range of 0 to 1
        scaling(X)

        #splitting data in train and test dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        
        if algo == 'SVM':
            model= SVC(kernel='linear') 
            model.fit(X_train,y_train)
            preds = model.predict(X_test)
            results(y_test, preds)
            
        elif algo == 'LR':
            model = LogisticRegression()
            result = model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results(y_test, preds)
            
        elif algo == 'XG':
            model = AdaBoostClassifier()
            result = model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results(y_test, preds)
            
        elif algo == 'KNN':
            model = KNeighborsClassifier(n_neighbors=3)
            result = model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results(y_test, preds)
        
    if algo_type == 'regression':
        #splitting data in labels and predictors
        y = dataset[label_col].values
        X = dataset.drop(columns = [label_col])
       
        #Scaling all the variables to a range of 0 to 1
        scaling(X)
        
        #splitting test and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if algo == 'LinearR':
            #splitting data in train and test dataset
            model = LinearRegression()
            result = model.fit(X_train, y_train)
            preds = model.predict(X_test)
            print(model.intercept_)
            print_evaluate(y_test, preds)
            plt.scatter(y_test, preds)


# Before applying function dataset need to be cleaned if contained any useless column or null data

# The below example perform Adaboost on given dataset. You can just change the name of alogrithm like (LR, SVM)

# Same above function is here perfroming linear regresion. Data need to be preprocessed before applying function

# # Function 16: Function to apply ANOVA and output results

# In[53]:


def anova_func(col1, col2, dataset):
    formula = '%s ~ C(Q("%s"))' %(col1, col2)
    model = ols(formula, data=dataset).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    display(anova_table)


# # Function 17: Function to generate correlation heatmaps

# In[57]:


def correlation_heatmap(dataset):
     sns.heatmap(sales.corr())


# # Function 18: Function to generate scatter plot

# In[66]:


def scatter_plot(dataset, col1, col2):
    sns.scatterplot(data=dataset, x=col1, y=col2)


# # Chi-square test function

# In[5]:


def chi_test(dataset, col1, col2):
    data_crosstab = pd.crosstab(dataset[col1], dataset[col2], margins = False) 
    print(data_crosstab) 

    stat, p, dof, expected = chi2_contingency(data_crosstab)
    print('dof=%d' % dof)
    print(expected)

    # interpret p-value
    alpha = 0.05
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')


# # Cross Validation

# In[ ]:


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


# # Validation metrics

# In[ ]:


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
        print("AUC Plot")
        #fpr, tpr, te_thresholds = roc_curve(testY, predictions)
        
       
        plt.plot(fpr , tpr)
        plt.plot([0,1],[0,1],'g--')
        plt.legend('TPR','FPR')
        plt.xlabel("True Positive Rate")
        plt.ylabel("False Positive Rate")
        plt.title("AUC(ROC curve)")
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        plt.show()
        
    
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


# In[ ]:


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


# In[ ]:



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


# In[ ]:


# Helper function to provide list of supported algorithms for Classification
def get_supported_algorithms():
    covered_algorithms = [LogReg, KNN, GadientBoosting, AdaBoost,
                          SVM, DecisionTree, RandomForest, NaiveBayes,
                          MultiLayerPerceptron]
    if XGBClassifier:
        covered_algorithms.append(XgBoost)
    if lgb:
        covered_algorithms.append(LightGbm)
    return covered_algorithms

# Helper function to provide list of supported algorithms for Regression
def get_supported_algorithms_reg():
    covered_algorithms = [LinearReg, RandomForestReg, PolynomialReg, SupportVectorRegression,
                          DecisionTreeReg, GradientBoostingReg, AdaBooostReg, VotingReg]
    return covered_algorithms


# In[ ]:


# Helper function to run all algorithms provided in algo_list over given dataframe, without cross validation
# By default it will run all supported algorithms 
def run_algorithms(df, label_col, algo_list=get_supported_algorithms(), feature_list=[]):
    """
    Run Algorithms with manual split
    
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
        


# In[ ]:


# With stratified kfold validation support
def run_algorithms_cv(df, label_col, algo_list=get_supported_algorithms(), feature_list=[], cross_valid_method=cross_valid_stratified_kf):
    """
    Run Algorithms with cross validation
    
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
        for trainX,trainY,testX,testY  in cross_valid_method(X, y, split=3):
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


# In[ ]:


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

#function for Lasso and ridge
def Lassofeatureimportance(df, trainX, testX, trainY, testY, random=None, regression=False):
    if regression:
        clf  = Lasso(alpha=0.1)

    else:
        clf = Lasso(alpha=0.1)
        
    lasso_coef = clf.fit(trainX,trainY).coef_
    
    #validationmetrics(clf,testX,testY)
    res = pd.Series(np.abs(lasso_coef),index=df.columns.values).sort_values(ascending=False)*100
    print(res)
    
    return res


# In[ ]:


# Helper function to select important features via RFFS, run supported ML algorithms over dataset with manual split and measure accuracy without Cross Validation - select features with importance >=threshold
def MachineLearningwithRFFS(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    res = RFfeatureimportance(df_cpy, trainX, testX, trainY, testY, regression=regression)
    
    impftrs = list(res[res > threshold].keys())
    #impftrs.append(label_col)
    
    print ("Selected Features =" + str(impftrs))
    print(df.shape)
    results = run_algorithms(df, label_col, algo_list=algo_list, feature_list=impftrs)
    return {"selected_features": impftrs, "results": results}


# Helper function to select important features via Lasso, run supported ML algorithms over dataset with manual split and measure accuracy without Cross Validation -
def MachineLearningwithLassoFS(df, label_col, algo_list=get_supported_algorithms(), regression=False):
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    res = Lassofeatureimportance(df_cpy, trainX, testX, trainY, testY, regression=regression)
    
    mean = np.mean(np.abs(res)) 
    impftrs = list(res[res > mean].keys())
    #impftrs.append(label_col)
    
    print ("Selected Features =" + str(impftrs))
    print(df.shape)
    results = run_algorithms(df, label_col, algo_list=algo_list, feature_list=impftrs)
    return {"selected_features": impftrs, "results": results}



# Helper function to select important features via RFFS, run supported ML algorithms over dataset with cross validation and measure accuracy --- select features with importance >=threshold
def MachineLearningwithRFFS_CV(df, label_col,threshold=5, algo_list=get_supported_algorithms(), regression=False):
    # lets create a copy of this dataframe and perform feature selection analysis over that
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
    

def MachineLearningwith_CV(df, label_col, algo_list=get_supported_algorithms(), regression=False):
    if regression:
        cross_valid_method = cross_valid_kfold
    else:
        cross_valid_method = cross_valid_stratified_kf
    results = run_algorithms_cv(df, label_col, algo_list=algo_list, cross_valid_method=cross_valid_method)
    return {"results": results}
    
# Addressing Class Imbalancing:
# Helper function to run all algorithms provided in algo_list over given dataframe, without cross validation
# By default it will run all supported algorithms 
def run_algorithms_cls_imb(df, label_col, algo_list=get_supported_algorithms(), feature_list=[]):
    """
    Run Algorithms with manual split
    
    """
    # Lets make a copy of dataframe and work on that to be on safe side 
    _df = df.copy()
    
    if feature_list:
        impftrs = feature_list
        impftrs.append(label_col)
        _df = _df[impftrs]
    
    _df, trainX, testX, trainY, testY = traintestsplit(_df, 0.2, 91, label_col=label_col)
    
     # class count
    class_count_0, class_count_1 = df[label_col].value_counts()

    # Separate class
    class_0 = df[df[label_col] == 0]
    class_1 = df[df[label_col] == 1]# print the shape of the class
    class_0_under = class_0.sample(class_count_1)
    test_under = pd.concat([class_0_under, class_1], axis=0)
    
    rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
    x_rus, y_rus = rus.fit_resample(trainX, trainY)

    algo_model_map = {}
    for algo in algo_list:
        print("============ " + algo.__name__ + " ===========")
        res = algo(x_rus, testX, y_rus, testY)
        algo_model_map[algo.__name__] = res.get("model_obj", None)
        print ("============================== \n")
    
    return algo_model_map
        




# With stratified kfold validation support
def run_algorithms__cls_imb_cv(df, label_col, algo_list=get_supported_algorithms(), feature_list=[], cross_valid_method=cross_valid_stratified_kf):
    """
    Run Algorithms with cross validation
    
    """
    _df = df.copy()
    X,y = XYsplit(_df, label_col)
      # class count
    class_count_0, class_count_1 = df[label_col].value_counts()

    # Separate class
    class_0 = df[df[label_col] == 0]
    class_1 = df[df[label_col] == 1]# print the shape of the class
    class_0_under = class_0.sample(class_count_1)
    test_under = pd.concat([class_0_under, class_1], axis=0)
    
    rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
    X, y = rus.fit_resample(X, y)

    
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

def MachineLearningwith_cls_imb_CV(df, label_col, algo_list=get_supported_algorithms(), regression=False):
    if regression:
        cross_valid_method = cross_valid_kfold
    else:
        cross_valid_method = cross_valid_stratified_kf
    results = run_algorithms__cls_imb_cv(df, label_col, algo_list=algo_list, cross_valid_method=cross_valid_method)
    return {"results": results}


def MachineLearningwith_cls_imb_RFFS(df, label_col, threshold=5, algo_list=get_supported_algorithms(), regression=False):
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    res = RFfeatureimportance(df_cpy, trainX, testX, trainY, testY, trees=10, regression=regression)
    
    impftrs = list(res[res > threshold].keys())
    #impftrs.append(label_col)
    
    print ("Selected Features =" + str(impftrs))
    print(df.shape)
    results = run_algorithms_cls_imb(df, label_col, algo_list=algo_list, feature_list=impftrs)
    return {"selected_features": impftrs, "results": results}


def MachineLearningwith_cls_imb_LassoFS(df, label_col, algo_list=get_supported_algorithms(), regression=False):
    # lets create a copy of this dataframe and perform feature selection analysis over that
    df_cpy = df.copy()
    df_cpy, trainX, testX, trainY, testY = traintestsplit(df_cpy, 0.2, 91, label_col=label_col)
    res = Lassofeatureimportance(df_cpy, trainX, testX, trainY, testY, regression=regression)
    
    mean = np.mean(np.abs(res)) 
    impftrs = list(res[res > mean].keys())
    #impftrs.append(label_col)
    
    print ("Selected Features =" + str(impftrs))
    print(df.shape)
    results = run_algorithms_cls_imb(df, label_col, algo_list=algo_list, feature_list=impftrs)
    return {"selected_features": impftrs, "results": results}




# Helper function to select important features via RFFS, run supported ML algorithms over dataset with cross validation and measure accuracy --- select features with importance >=threshold
def MachineLearningwith_cls_imb_RFFS_CV(df, label_col,threshold=5, algo_list=get_supported_algorithms(), regression=False):
    # lets create a copy of this dataframe and perform feature selection analysis over that
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
    results = run_algorithms__cls_imb_cv(df, label_col, algo_list=algo_list, feature_list=impftrs, cross_valid_method=cross_valid_method)
    return {"selected_features": impftrs, "results": results}
    







# Helper function to select important features via REFS, run supported ML algorithms over dataset with manual split and measure accuracy, with CV ... select features with importance >=threshold
# flexible enough to use any algorithm for recursive feature elimination and any alogorithm to run on selected features
def GenericREFS_CV(df, label_col,
                algo_list=get_supported_algorithms(),
                regression=False,
                re_algo=RandomForestClassifier,
                **kwargs):
    
    X,y = XYsplit(df, label_col)
    clf = re_algo(**kwargs)
    selector = RFECV(estimator=clf, step=1, cv=10)
    selector = selector.fit(X,y)
    feature_list = X.columns[selector.support_].tolist()
    if regression:
        cross_valid_method = cross_valid_kfold
    else:
        cross_valid_method = cross_valid_stratified_kf
    results = run_algorithms_cv(df, label_col, algo_list=algo_list, feature_list=feature_list, cross_valid_method=cross_valid_method)
    return {"selected_features": feature_list, "results": results}

# Helper function to provide list of classification algorithms to be used for recursive elimination feature selection
def get_supported_algorithms_refs():
    algo_list = [LogisticRegression, GradientBoostingClassifier, AdaBoostClassifier,
                          DecisionTreeClassifier, RandomForestClassifier]
    return algo_list

# Helper function to provide list of regression algorithms to be used for recursive elimination feature selection
def get_supported_reg_algorithms_refs():
    algo_list = [LinearRegression, RandomForestRegressor,
                 DecisionTreeRegressor, GradientBoostingRegressor, AdaBoostRegressor]
    return algo_list

