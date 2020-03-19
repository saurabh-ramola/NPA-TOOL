#!/usr/bin/env python
# coding: utf-8

# In[62]:


from collections import Counter

import _pickle as pickle
import xgboost as xgb
import datetime
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd
import os
import math
import numpy as np
from collections import defaultdict
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score,recall_score,confusion_matrix
from sklearn import tree
from xgboost import XGBClassifier, plot_tree
import xgboost
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold

import collections
global flag
flag = True
class OrderedSet(collections.Set):
	def __init__(self, iterable=()):
		self.d = collections.OrderedDict.fromkeys(iterable)

	def __len__(self):
		return len(self.d)

	def __contains__(self, element):
		return element in self.d

	def __iter__(self):
		return iter(self.d)



category = ['isChildrenGoingToSchool', 'Latest_House_Electricity_Payment', 'House_Ownership', 'Owned_Details', 'isClient_LivingIn_SameHouse_AsPer_LoanApplication', 'Original_Client_IdentityProof', 'Original_Client_AddressProof', 'Original_Client_OwnershipProof', 'Original_Client_UtilityBillsProof', 'Original_Guarantor_IdentityProof', 'Original_Guarantor_AddressProof', 'Business_Relocation_Risk', 'Business_ManagedBy', 'isBorrower_having_valid_BusinessDocuments', 'Level_Of_Formalisation', 'Market_Reputation', 'Social_Reputation', 'Latest_Shop_Electricity_Payment', 'Housekeeping_Of_BusinessPremises', 'Business_PurchaseRecords', 'Business_SalesRecords', 'Preferred_RepaymentMode', 'BusinessSector', 'BusinessPremise_Ownership', 'BusinessLocation', 'LoanPurpose', 'BankingTransactions', 'CreditBeureau_History', 'Appraisal_SourcedFrom', 'TypeOfApplication', 'TypeOfBusiness', 'GuarantorProfessional', 'Submission', 'TypeOfGuarantor', 'F1SEX', 'F1EDUCATION', 'F2SEX', 'F2EDUCATION', 'F3SEX', 'F3EDUCATION', 'RELATIONTYPE', 'MARITAL_STATUS_CODE', 'STATE', 'CUST_TYPE', 'OCCUPATION', 'CASTE', 'ANNUAL_INCOME']
loanPerf_file = 'loan_performance_2.csv'


def split_X_y(dfOnes,dfZeros):
	X = []
	Y = []
	ids = []
	
	yvals = 1
	df = dfOnes.drop(['DEFAULT_DUMMY',0],axis=1)
	idDf = dfOnes[0]
	
	for i in range(len(dfOnes)):
		y = yvals
		ids.append(idDf.iloc[i])
		
		feat_vect = [0 if math.isnan(y) else y for y in list(df.iloc[i])]
		
		X.append(feat_vect)
		Y.append(y)
		
	yvals = 0
	df = dfZeros.drop(['DEFAULT_DUMMY',0],axis=1)
	idDf = dfZeros[0]
	
	for i in range(len(dfZeros)):
		y = 0
		ids.append(idDf.iloc[i])
		
		feat_vect = [0 if math.isnan(y) else y for y in list(df.iloc[i])]
		
		X.append(feat_vect)
		Y.append(y)
	
	return np.array(X),np.array(Y),ids

def save_app_data(file,file1,file2):
	"""
	Append to data/unlabelled_data.csv
	"""
	appraisal_data = pd.read_csv(file)
	application_data = pd.read_csv(file1)
	loanPerf_data = pd.read_csv(file2)

	appr_loanPerf_data = pd.merge(appraisal_data, loanPerf_data, left_on='ApplicationId', right_on='APPLICATIONID')

	appr_loanPerf_data = pd.merge(appr_loanPerf_data,application_data, left_on='ApplicationId', right_on='ApplicationId')

	for key in appr_loanPerf_data:
		try:
			appr_loanPerf_data[key] = appr_loanPerf_data[key].str.lower()
		except:
			pass

	mappings = {}
	toDrop = ['NPA_DUMMY','CIBIL']
	for key in appr_loanPerf_data:
		try:
			appr_loanPerf_data[key] = pd.to_numeric(appr_loanPerf_data[key])
			continue
		
		except:
			uniqs = appr_loanPerf_data[key].unique()

			keymap = defaultdict(lambda:-1)
			for i in range(len(uniqs)):
				keymap[uniqs[i]] = i

			mappings[key] = keymap
			
			if len(keymap) > 15:
				toDrop.append(key)

	toDrop = toDrop + ['SubDistrict',
	 'subkcreditscore',
	 'GuarantorCibil',
	 'HouseLoc_Latitude',
	 'HouseLoc_Longitude',
	 'BusinessLoc_Latitude',
	 'BusinessLoc_Longitude',
	 'Guarantor_Latitude',
	 'Guarantor_Longitude','YEAR_ENT']

	appr_loanPerf_data = appr_loanPerf_data.drop(toDrop,axis=1)
	appr_loanPerf_data = appr_loanPerf_data.fillna(0)

	appr_loanPerf_data.to_csv('./encodingData.csv',index=False)
	categoricals = appr_loanPerf_data.select_dtypes(include=[object])

	dfs = appr_loanPerf_data.drop(['DEFAULT_DUMMY'],axis=1)
	dfs.to_csv('./data.csv',index=False)
	categoricals = categoricals.apply(lambda col: col.astype(str), axis=0, result_type='expand')

	numericals = list(OrderedSet(appr_loanPerf_data) - OrderedSet(categoricals))
	numericals = appr_loanPerf_data[numericals].drop(['DEFAULT_DUMMY'],axis=1)
	numericalArray = numericals.to_numpy()
	enc = OneHotEncoder()
	categoricalOneHot = enc.fit_transform(categoricals).toarray()
	X = np.concatenate((numericalArray,categoricalOneHot),axis=1)
	Y = appr_loanPerf_data['DEFAULT_DUMMY'].to_numpy()

	processedData = pd.DataFrame(X)
	processedData['DEFAULT_DUMMY'] = Y

	gk = processedData.groupby('DEFAULT_DUMMY')

	X, Y, ids = split_X_y(gk.get_group(1),gk.get_group(0))
	length = X.shape[1]
	# print(type(Y))
	train(X,Y)

def max_value(arr):
	if arr[0] >= arr[1]:
		return arr[0]
	else:
		return arr[1]
def return_applicant(df):
	appr_loanPerf_data = pd.read_csv('./encodingData.csv')
	
	categoricals = appr_loanPerf_data.select_dtypes(include=[object])

	categoricals = categoricals.apply(lambda col: col.astype(str), axis=0, result_type='expand')
	numericals = list(OrderedSet(appr_loanPerf_data) - OrderedSet(categoricals))
	numericals = df[numericals].drop(['DEFAULT_DUMMY'],axis=1)
	cat = df[list(OrderedSet(categoricals))]
	cat = cat.apply(lambda col: col.astype(str), axis=0, result_type='expand')
	numericalArray = numericals.to_numpy()

	enc = OneHotEncoder().fit(categoricals)
	oneHot = enc.transform(cat).toarray()
	X = np.concatenate((numericalArray,oneHot),axis=1)

	if len(os.listdir('./model2')) != 0:
		flag = 0
		return X,flag 
	else:
		x = pd.DataFrame(X).drop([0],axis=1).to_numpy()
		print(x.shape)
		flag = 1
		return x,flag

def train(X,Y):
	model=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
			  colsample_bynode=1, colsample_bytree=0.8, gamma=1,
			  learning_rate=0.02, max_delta_step=0, max_depth=5,
			  min_child_weight=5, missing=None, n_estimators=600, n_jobs=1,
			  nthread=1, objective='binary:logistic', random_state=0,
			  reg_alpha=0, reg_lambda=1, scale_pos_weight=5, seed=None,
			  silent=True, subsample=0.8, verbosity=1)

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	print("Training : ",X_train.shape, y_train)
	model.fit(X_train, y_train)
	
	print('saving ')
	if len(os.listdir('./model2')) == 0 and len(os.listdir('./model1')) == 0:
		pickle.dump(model, open('./model1/model.pkl', 'wb+'))
	elif len(os.listdir('./model2')) == 0 and len(os.listdir('./model1')) != 0:
		pickle.dump(model, open('./model2/model.pkl', 'wb+'))



def update_default_data(npa_file):
	appr_loanPerf_data = pd.read_csv('./encodingData.csv')
	df = pd.read_csv(npa_file)
	appr_loanPerf_data = pd.concat([appr_loanPerf_data,df])
	appr_loanPerf_data.to_csv('./encodingData.csv',index=False)
	categoricals = appr_loanPerf_data.select_dtypes(include=[object])

	categoricals = categoricals.apply(lambda col: col.astype(str), axis=0, result_type='expand')
	numericals = list(OrderedSet(appr_loanPerf_data) - OrderedSet(categoricals))
	numericals = appr_loanPerf_data[numericals].drop(['DEFAULT_DUMMY'],axis=1)
	numericalArray = numericals.to_numpy()

	enc = OneHotEncoder(handle_unknown='ignore')
	categoricalOneHot = enc.fit_transform(categoricals).toarray()

	X = np.concatenate((numericalArray,categoricalOneHot),axis=1)
	Y = appr_loanPerf_data['DEFAULT_DUMMY'].to_numpy()
	length = X.shape[1]
	print(len(X))
	train(X,Y) 

def get_stats():
	loan_file = len(open('./data/loan_performance_2.csv').read().split('\n')) - 2
	# unlabelled = len(open('./data/unlabelled_data.csv').read().split('\n')) - 2
	return loan_file



