import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
import xlsxwriter

dataset = pd.read_excel('Denguedatasample1.xlsx')

def getmean(array):
    mean = 0
    for data in array:
        mean+=data
    mean=mean/10    
    return mean

#print dataset to check if it reads the correct dataset
#comment this line if you are getting the correct dataset
#print(dataset)
#separating the features and output values (target)
#initializing the target
target = []
#initializing the features
features = []
for i in range(len(dataset.columns)):
    if i < len(dataset.columns)-1:
        features.append(dataset.columns[i])
    else:
        target.append(dataset.columns[i])


#checking if you have separated them correctly
#comment the next two lines if you are done checking
#print(features)
#print(target)
X = dataset[features] 

Y = dataset.FINAL

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42, train_size = 0.80)

kf = StratifiedKFold(n_splits=10,shuffle=True, random_state=4)

clf = RandomForestClassifier(
        criterion='gini',
        n_estimators=110,
        max_depth=4,
        random_state=42
    )

scorer ={'A':'accuracy',
          'P':'precision',
          'R':'recall',
          'S':make_scorer(recall_score,pos_label=0),
          'F':'f1',
          'AUC':'roc_auc'}


model = [clf]
    

model_name = ["Random Forest"]

#list and call all scorers
scorer ={'A':'accuracy',
        'P':'precision',
        'R':'recall',
        'S':make_scorer(recall_score,pos_label=0),
        'F':'f1',
        'AUC':'roc_auc'}

#testing the performance of the chosen model using cross-validation
arr = []
for ind,m in enumerate(model):
    arr1 = []
    print("Currently Running: "+model_name[ind])
    scores = cross_validate(m, X, Y, cv=kf, scoring=scorer,return_train_score=False)
    #append the scores according to the scorer
    arr1.append(scores['test_A']);
    arr1.append(scores['test_P'])
    arr1.append(scores['test_R'])
    arr1.append(scores['test_S'])
    arr1.append(scores['test_F'])
    arr1.append(scores['test_AUC'])
    arr.append(arr1)
    
    # Use predict_proba to predict probability of the class
    model1 = m.fit(X_train, Y_train)
    y_pred = m.predict_proba(X_test)[:,1]

    #from plot_metric.functions import BinaryClassification
    # Visualisation with plot_metric
    #bc = BinaryClassification(Y_test, y_pred, labels=["Class 1"], threshold = 0.5)

    # Figures
    #plt.figure(figsize=(5,5))
    #bc.plot_roc_curve()
    #plt.show()
    
print("Finished Testing all Models")

workbook = xlsxwriter.Workbook(f"Mello_roman_data_result_1_components.xlsx")
worksheet = workbook.add_worksheet()

#change the elements inside if you change the value of f
header = [""] + [f"Fold {i}" for i in range(1, 11)] + ["Average/Mean"]
#change the elements inside if you change the elements in the scorer
scorename = ["Accuracy","Precision","Recall","Specificity","F1 Score","AUC"]
row = 1
#printing the elements in header at the first row in the worksheet
worksheet.write_row(0,0,header)
auc = []

for index,i in enumerate(arr):
    n= "Result For "+model_name[index]
    data = i
    worksheet.write(row, 0,n)
    row+=1
    for a,b in enumerate(data):
        worksheet.write(row,0,scorename[a]) 
        worksheet.write_row(row,1,b)
        worksheet.write(row,10+1,getmean(b))
        row+=1
        if a == len(data)-1:
            auc.append(getmean(b))
    #calculating the true positive rate     
    worksheet.write(row,0,"TPR")
    TPR = []
    for d1 in range(10):
        #TPR is just equal to the recall rate (data[2]), thus the copying
        #change the index when you change the position of recall in the scorer
        #change the formula if you get rid of the recall in the scorer
        worksheet.write(row,d1+1,data[2][d1])
        TPR.append(data[2][d1])
    #their mean is also equal   
    tprmean = getmean(data[2])
    worksheet.write(row,10+1,tprmean)
    row+=1
    #calculating the false positive rate
    worksheet.write(row,0,"FPR")
    fpr = []
    FPR = []
    #FPR is the reverse of specificity score thus the 1-specificity formula
    for d2 in range(10):
        worksheet.write(row,d2+1,(1-data[3][d2]))
        fpr.append((1-data[3][d2]))
        FPR.append(1-data[3][d2])
    #solving for the mean of FPR    
    fprmean = getmean(fpr)
    worksheet.write(row,10+1,fprmean)
    row+=1
#closing the workbook    
workbook.close()

#Best Hyperparameters: {'n_estimators': 110, 'max_depth': 4}
#Best Accuracy: 0.6666549109504497
