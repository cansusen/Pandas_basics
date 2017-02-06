import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

def draw_graph(los,data):
    plt.plot(item)
    plt.axis([0, los, -3, 3])
    plt.xlabel('Time')
    plt.ylabel('f(x)')
    plt.show()
    
def draw_stacked_graph(val1, val2):
    N = 60
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, val1, width, color='r')
    p2 = plt.bar(ind, val2, width, color='b', bottom=val1)

    plt.ylabel('Frequency')
    plt.xlabel('Svm Predictions')
    plt.title('The continuous output of the SVM')
    #plt.xticks(arange(7), ('-3', '-2', '-1', '0', '1', '2', '3'))
    #xlabels = ['-3', '-2', '-1', '0', '1', '2', '3']
    #plt.xticks(range(-300, 300, 10),xlabels)
    #plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Positive', 'Negative'))

    plt.show()
    
def between(l1,low,high):
    l2 = []
    for i in l1:
        if(i > low and i < high):
            l2.append(i)
    return l2


wanna_see_graphs = 0


df_pos = pd.read_csv('input_feature_file.csv',sep=',')
df_neg = pd.read_csv('input_feature_file_negative.csv',sep=',')
df     = df_pos.append(df_neg, ignore_index=True)

df = df.fillna(df.mode().iloc[0])           #If missing, fill with mode value

#Seperate the target value and the features
label       = df['label']
day_id      = df['day_id']
hadm_id     = df['hadm_id']
len_of_stay = df['los']
columns_to_drop = ['label', 'day_id', 'hadm_id']
features_temp = df.drop(columns_to_drop,1)

#change categorical variables into binary variables
categorical_features = ["admission_type","admission_location","discharge_location","insurance","language","religion","marital_status","ethnicity"]
categorized_data     = pd.get_dummies(features_temp[categorical_features])

#remove the categorical variables form the dataset and add the binary variables to it.
features_temp = features_temp.drop(categorical_features,1)
features      = pd.concat([features_temp,categorized_data],axis=1,join_axes=[features_temp.index])

print(len(features.columns))
features.to_csv('feature_vectors.csv')

#Different Scaling methods for data, use only one
feat_minmax   = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(features).transform(features)
feat_standard = preprocessing.StandardScaler().fit(features).transform(features)
feat_norm     = preprocessing.Normalizer().fit(features).transform(features)

#Feature selection methods
#Method 1: If a feature mostly the same for all instances, remove it. 
features_highvar = VarianceThreshold(threshold=(.8 * (1 - .8))).fit_transform(features)

#Method 2: Pick best K features
features_Kbest = SelectKBest(chi2, k=30).fit_transform(feat_norm, label)
'''
Start classification process
'''
# split the dataset in training and test set:
X_train, X_test, y_train, y_test = train_test_split(features_Kbest, label, test_size=0.3, random_state=42)




'''
Cross-Validation
'''

param_grid = [
  {'C': [0.01, 0.1, 1, 10]},
 ]

print("# Tuning hyper-parameters")
clf = GridSearchCV(LinearSVC(), param_grid, cv=5)
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()

model = SelectFromModel(clf, prefit=True)

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()


'''
Tranining and Testing
'''
'''
#classifier = svm.SVC(kernel='linear', C=0.01)
#y_pred = classifier.fit(X_train, y_train).predict(X_test)

#clf = SVC(C=1000, kernel = 'linear')
clf = LinearSVC(C=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("assigned weights to features: ")
print(clf.coef_)
#target_names = ['','']
print(classification_report(y_test, y_pred)) #target_names=target_names
'''

'''
Save classifier
'''
#joblib.dump(clf, 'classifier_cdiff.pkl')

'''
Get distance from the hyperplane
'''
distance_from_boundry = clf.decision_function(X_test)
#print(distance_from_boundry)





if (wanna_see_graphs == 1):
    '''
    Stacked graph of results
    '''
    distances_1s = [ i for (i,j) in zip(distance_from_boundry,y_test) if j >0 ]
    distances_0s = [ i for (i,j) in zip(distance_from_boundry,y_test) if j <=0 ]
    
    graph_val_neg = []
    graph_val_pos = []
    for i in [float(j) / 100 for j in range(-300, 300, 10)]:
        graph_val_neg.append(len(between(distances_0s,i,i+0.1)))
        graph_val_pos.append(len(between(distances_1s,i,i+0.1)))
        #print( len(between(distances_1s,i,i+0.1)) )
        #print( len(between(distances_0s,i,i+0.1)) )
        
    
    draw_stacked_graph(graph_val_pos, graph_val_neg)

    '''
    Process all risk_processes
    '''
    all_risk_scores = []
    patient_risk_scores = []
    
    prev_hadmid = hadm_id[0]
    patient_risk_scores.append(distance_from_boundry[0])
    
    for a,b in zip(hadm_id[1:], distance_from_boundry[1:]):
        if (prev_hadmid == a):
            patient_risk_scores.append(b)
        else:
            all_risk_scores.append(patient_risk_scores)
            patient_risk_scores = []
            
            prev_hadmid=a
            patient_risk_scores.append(b)
        
        
    #We can draw the output time series for risk processes of patients
    for los,item in zip(len_of_stay[1:5],all_risk_scores[1:5]):
        print('len of stay: ' , los)
        print('risk process: ', item)
        draw_graph(los,item)
        
    '''    
    Calculate the phi for admissions
    '''
    phi = []
    for admission in all_risk_scores:
        #each admission is a risk process for an admission i.e. time series of risk scores for each day
        tmp = np.mean(abs(np.diff(admission)))
        phi.append(tmp)
    
    #print(phi)
    
