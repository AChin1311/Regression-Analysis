import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn import metrics
#from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#from sklearn.feature_selection import f_regression, mutual_info_regression
#import matplotlib.pyplot as plt
import time




""" ====== Prepare data ====== """

content = pd.read_csv("network_backup_dataset.csv")
content = content.replace({'Day of Week': {'Monday' : 0, 'Tuesday' : 1, 'Wednesday' : 2 ,
                                         'Thursday' : 3, 'Friday' : 4,
                                         'Saturday' : 5, 'Sunday' : 6 }})

WorkFlow_list = sorted(pd.unique(content['Work-Flow-ID']))
File_list     = sorted(pd.unique(content['File Name']))  
num_WorkFlow  = len(WorkFlow_list)
num_File      = len(File_list)

for i in np.arange(num_WorkFlow):
    content = content.replace({'Work-Flow-ID': {'work_flow_%d'%i:i}})

for i in np.arange(num_File):
    content = content.replace({'File Name': {'File_%d'%i:i}})


cont_np = content.values

X_list = []
Y_list = []

for i in np.arange(4):
    cont_list_i = cont_np[np.where(cont_np[:,3] == i)]
    X_i = np.delete(cont_list_i[: , 0:5],3,axis=1)
    Y_i = cont_list_i[: ,  5 ]
    X_list.append(X_i)
    Y_list.append(Y_i)



#X_org = content_all_np[: , 0:5]
#Y_org = content_all_np[: ,  5 ]


del WorkFlow_list,File_list,cont_np, num_WorkFlow,num_File,i,X_i,Y_i,cont_list_i
# delete unused variables





def hot16comb(X):

    # the 16 combinations, split X into 5 columns
    enc1 =  preprocessing.OneHotEncoder(sparse = False)
    num_sample = X.shape[0]
    
    X0 = X[:,0].reshape((num_sample,1))
    X1 = X[:,1].reshape((num_sample,1))
    X2 = X[:,2].reshape((num_sample,1))
    X3 = X[:,3].reshape((num_sample,1))

    
    X_Hot0 = enc1.fit_transform(X0)
    X_Hot1 = enc1.fit_transform(X1)
    X_Hot2 = enc1.fit_transform(X2)
    X_Hot3 = enc1.fit_transform(X3)
    
    X_Very_Hot = []
    Hot_comb   = []
    
    for i in np.arange(16):
    
        a = str(bin(i))[2:].rjust(4,'0')
            
        H0 = X0*int(a[0]) + X_Hot0*(1-int(a[0]))
        H1 = X1*int(a[1]) + X_Hot1*(1-int(a[1]))
        H2 = X2*int(a[2]) + X_Hot2*(1-int(a[2]))
        H3 = X3*int(a[3]) + X_Hot3*(1-int(a[3]))
        
        X_Very_Hot_i = np.concatenate((H0,H1,H2,H3),axis =1)
        X_Very_Hot.append(X_Very_Hot_i)
        Hot_comb.append(a)
    
    print("X_Very_Hot generated")
    return X_Very_Hot,Hot_comb


X_Very_Hot = []
Hot_comb   = []
for i in np.arange(4):

    X_Very_Hot_i,Hot_comb_i = hot16comb(X_list[i])
    X_Very_Hot.append(X_Very_Hot_i)
    Hot_comb.append(Hot_comb_i)

del X_Very_Hot_i,Hot_comb_i


def hot16acc_best(X_Very_Hot,Hot_comb,Y,reg_model,alpha,l1_ratio):
    avg_test_RMSE_i  = 0
    avg_train_RMSE_i = 0
    best_RMSE_test   = 0
    best_RMSE_train  = 0
    best_comb        = 0
    for i in np.arange(16):
        X_in = X_Very_Hot[i]
        
        if reg_model   == 'lin_reg':
            avg_test_RMSE_i,avg_train_RMSE_i = kfold_lin_reg(X_in,Y)
        elif reg_model == 'ridg':
            avg_test_RMSE_i,avg_train_RMSE_i = kfold_ridg(X_in,Y,alpha)
        elif reg_model == 'lass':
            avg_test_RMSE_i,avg_train_RMSE_i = kfold_lass(X_in,Y,alpha)
        elif reg_model == 'elst':
            avg_test_RMSE_i,avg_train_RMSE_i = kfold_elst(X_in,Y,alpha,l1_ratio)
        else:
            print("Wrong Input")
            break
        
        name = Hot_comb[i]
        
        if i ==0:
            best_RMSE_test   = avg_test_RMSE_i
            best_comb        = name
            best_RMSE_train  = avg_train_RMSE_i
        
        if avg_test_RMSE_i < best_RMSE_test:
            best_RMSE_test   = avg_test_RMSE_i
            best_comb        = name
            best_RMSE_train  = avg_train_RMSE_i

        
    return best_RMSE_test,best_RMSE_train,best_comb



def kfold_ridg(X,Y,alpha):
    test_rmse  = []
    train_rmse = []
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    fold_i =0
    for train_id, test_id in kf.split(X):
        fold_i += 1
        X_train, X_test = X[train_id], X[test_id]
        Y_train, Y_test = Y[train_id], Y[test_id]

        model_1 = linear_model.Ridge(alpha)
        model_1.fit(X_train,Y_train)

        Y_test_pred  = model_1.predict(X_test)
        Y_train_pred = model_1.predict(X_train)

        test_rmse.append ( np.sqrt( metrics.mean_squared_error(Y_test , Y_test_pred )))
        train_rmse.append( np.sqrt( metrics.mean_squared_error(Y_train, Y_train_pred)))

    avg_test_RMSE   = sum(test_rmse)/10
    avg_train_RMSE  = sum(train_rmse)/10

    return avg_test_RMSE,avg_train_RMSE



def kfold_lass(X,Y,alpha):
    test_rmse  = []
    train_rmse = []
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    fold_i =0
    for train_id, test_id in kf.split(X):
        fold_i += 1
        X_train, X_test = X[train_id], X[test_id]
        Y_train, Y_test = Y[train_id], Y[test_id]

        model_1 = linear_model.Lasso(alpha)
        model_1.fit(X_train,Y_train)

        Y_test_pred  = model_1.predict(X_test)
        Y_train_pred = model_1.predict(X_train)

        test_rmse.append ( np.sqrt( metrics.mean_squared_error(Y_test , Y_test_pred )))
        train_rmse.append( np.sqrt( metrics.mean_squared_error(Y_train, Y_train_pred)))

    avg_test_RMSE   = sum(test_rmse)/10
    avg_train_RMSE  = sum(train_rmse)/10

    return avg_test_RMSE,avg_train_RMSE




def kfold_elst(X,Y,alpha,l1_ratio):
    test_rmse  = []
    train_rmse = []
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    fold_i =0
    for train_id, test_id in kf.split(X):
        fold_i += 1
        X_train, X_test = X[train_id], X[test_id]
        Y_train, Y_test = Y[train_id], Y[test_id]

        model_1 = linear_model.ElasticNet(alpha,l1_ratio)
        model_1.fit(X_train,Y_train)

        Y_test_pred  = model_1.predict(X_test)
        Y_train_pred = model_1.predict(X_train)

        test_rmse.append ( np.sqrt( metrics.mean_squared_error(Y_test , Y_test_pred )))
        train_rmse.append( np.sqrt( metrics.mean_squared_error(Y_train, Y_train_pred)))

    avg_test_RMSE   = sum(test_rmse)/10
    avg_train_RMSE  = sum(train_rmse)/10

    return avg_test_RMSE,avg_train_RMSE




print("\n")
""" === 1. Ridge === """
print("=== 1. Ridge ===")
l1_ratio =0
alpha_all = np.array([0.1,0.3,0.5,1,3,5,10,20,40,80,120])

RMSE_ridge ={}

for wf in np.arange(4):

    RMSE_ridge_wf = np.zeros((len(alpha_all),4))
    i=0
    for alpha in alpha_all:
    
        best_RMSE_test_i,best_RMSE_train_i,best_comb_i = hot16acc_best(X_Very_Hot[wf],Hot_comb[wf],Y_list[wf],'lin_reg',alpha,l1_ratio)
    
        RMSE_ridge_wf[i,0] = alpha
        RMSE_ridge_wf[i,1] = best_comb_i
        RMSE_ridge_wf[i,2] = best_RMSE_test_i
        RMSE_ridge_wf[i,3] = best_RMSE_train_i
        i += 1        
        
    RMSE_column = RMSE_ridge_wf[:,2]
    RMSE_ind    = np.argmin(RMSE_column)
    best_alpha  = RMSE_ridge_wf[RMSE_ind,0]
    best_comb   = str(int(RMSE_ridge_wf[RMSE_ind,1])).rjust(4,'0')
    best_test_err  = RMSE_ridge_wf[RMSE_ind,2]
    avg_train_err  = RMSE_ridge_wf[RMSE_ind,3]

    RMSE_ridge['work_flow_%d'%wf] = RMSE_ridge_wf
    
    print("RMSE_ridge generated wf = ", wf)
    print("Best test error = ",best_test_err,
          "\nAvg train error = ",avg_train_err,
          "\nalpha = ",best_alpha,"\ncombination = ",best_comb,"\n")
        
    

print("Part 1 completed")



#RMSE_ridge generated wf =  0
#Best test error =  0.025518351854589373 
#Avg train error =  0.025519732378102705 
#alpha =  0.1 
#combination =  1001 
#
#RMSE_ridge generated wf =  1
#Best test error =  0.11239622690453732 
#Avg train error =  0.1134477906908699 
#alpha =  0.1 
#combination =  1001 
#
#RMSE_ridge generated wf =  2
#Best test error =  0.030898544174410524 
#Avg train error =  0.03096898087332387 
#alpha =  0.1 
#combination =  1001 
#
#RMSE_ridge generated wf =  3
#Best test error =  0.005271988971669496 
#Avg train error =  0.005300703767936993 
#alpha =  0.1 
#combination =  1000 






print("\n")
""" === 2. Lasso === """
print("=== 2. Lasso ===")
l1_ratio =1




alpha_all = np.array([0.1,0.3,0.5,1,3,5,10,20,40,80,120])

RMSE_lass ={}

for wf in np.arange(4):

    RMSE_wf = np.zeros((len(alpha_all),4))
    i=0
    for alpha in alpha_all:
    
        best_RMSE_test_i,best_RMSE_train_i,best_comb_i = hot16acc_best(X_Very_Hot[wf],Hot_comb[wf],Y_list[wf],'lass',alpha,l1_ratio)
    
        RMSE_wf[i,0] = alpha
        RMSE_wf[i,1] = best_comb_i
        RMSE_wf[i,2] = best_RMSE_test_i
        RMSE_wf[i,3] = best_RMSE_train_i
        i += 1        
        
    RMSE_column = RMSE_wf[:,2]
    RMSE_ind    = np.argmin(RMSE_column)
    best_alpha  = RMSE_wf[RMSE_ind,0]
    best_comb   = str(int(RMSE_wf[RMSE_ind,1])).rjust(4,'0')
    best_test_err  = RMSE_wf[RMSE_ind,2]
    avg_train_err  = RMSE_wf[RMSE_ind,3]
    
    RMSE_lass['work_flow_%d'%wf] = RMSE_wf
    
    print("RMSE_lass generated wf = ", wf)
    print("Best test error = ",best_test_err,
          "\nAvg train error = ",avg_train_err,
          "\nalpha = ",best_alpha,"\ncombination = ",best_comb,"\n")
        
    


print("Part 2 completed")


#RMSE_lass generated wf =  0
#Best test error =  0.04630552033765081 
#Avg train error =  0.046166271086371405 
#alpha =  0.1 
#combination =  0010 
#
#RMSE_lass generated wf =  1
#Best test error =  0.15554034942723133 
#Avg train error =  0.1575251500370659 
#alpha =  0.1 
#combination =  0100 
#
#RMSE_lass generated wf =  2
#Best test error =  0.04352574329265917 
#Avg train error =  0.04385087394814184 
#alpha =  0.1 
#combination =  0000 
#
#RMSE_lass generated wf =  3
#Best test error =  0.0072542722908727235 
#Avg train error =  0.0073111657901835676 
#alpha =  0.1 
#combination =  0000 


print("\n")
""" === 3. Elastic === """
print("=== 3. Elastic ===")


alpha_all     = np.array([0.5,5.0,20.0,80.0,120.0,200.0])
l1_ratio_all  = np.array([0.01,0.1,0.3,0.5,0.7,0.9,0.99])


RMSE_elst     = {}
RMSE_elst_map = {}

for wf in np.arange(4):

    RMSE_wf = np.zeros((len(alpha_all)*len(l1_ratio_all),5))
    RMSE_map_wf  = np.zeros((len(alpha_all)+1, len(l1_ratio_all)+1 ))
    RMSE_map_wf[0,1:] = l1_ratio_all
    RMSE_map_wf[1:,0] = alpha_all

    i=0
    k=0
    for alpha in alpha_all:
        j=0
        t1 = time.time()
        for l1_ratio in l1_ratio_all:
            best_RMSE_test_i,best_RMSE_train_i,best_comb_i = hot16acc_best(X_Very_Hot[wf],Hot_comb[wf],Y_list[wf],'elst',alpha,l1_ratio)
        
            RMSE_wf[k,0] = alpha
            RMSE_wf[k,1] = best_comb_i
            RMSE_wf[k,2] = best_RMSE_test_i
            RMSE_wf[k,3] = best_RMSE_train_i
            RMSE_wf[k,4] = l1_ratio
            
            
            RMSE_map_wf[i+1,j+1]  = best_RMSE_test_i
            j += 1
            k += 1
        t2 = time.time()
#        print("alpha = ",alpha," done")
#        print("Time consumed:",t2-t1)
        i +=1


    RMSE_elst['work_flow_%d'%wf]     = RMSE_wf
    RMSE_elst_map['work_flow_%d'%wf] = RMSE_map_wf

    RMSE_column = RMSE_wf[:,2]
    RMSE_ind    = np.argmin(RMSE_column)
    best_alpha  = RMSE_wf[RMSE_ind,0]
    best_comb   = str(int(RMSE_wf[RMSE_ind,1])).rjust(4,'0')
    best_test_err  = RMSE_wf[RMSE_ind,2]
    avg_train_err  = RMSE_wf[RMSE_ind,3]
    best_l1_ratio  = RMSE_wf[RMSE_ind,4]


    print("\nRMSE_elst generated wf = ", wf)
    print("Best test error = ",best_test_err,
          "\nAvg train error = ",avg_train_err,
          "\nalpha = ",best_alpha,"\nl1_ratio = ", best_l1_ratio,
          "\ncombination = ",best_comb,"\n")



print("Part 3 completed")


del RMSE_column,RMSE_ind,best_alpha,best_comb,best_test_err,avg_train_err,best_l1_ratio
del i,j,k,wf,t1,t2.alpha,l1_ratio,RMSE_wf,RMSE_map_wf
del best_RMSE_test_i,best_RMSE_train_i,best_comb_i


#RMSE_elst generated wf =  0
#Best test error =  0.035994874215625666 
#Avg train error =  0.035962452124660135 
#alpha =  0.5 
#l1_ratio =  0.01 
#combination =  1110 
#
#RMSE_elst generated wf =  1
#Best test error =  0.1451475645446 
#Avg train error =  0.14690145819024197 
#alpha =  0.5 
#l1_ratio =  0.01 
#combination =  0100 
#
#RMSE_elst generated wf =  2
#Best test error =  0.04244772793140494 
#Avg train error =  0.04275135890739546 
#alpha =  0.5 
#l1_ratio =  0.01 
#combination =  0010 
#
#RMSE_elst generated wf =  3
#Best test error =  0.0072542722908727235 
#Avg train error =  0.0073111657901835676 
#alpha =  0.5 
#l1_ratio =  0.01 
#combination =  0000 









