from __future__ import absolute_import
from __future__ import print_function

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from mimic3benchmark.readers import PhenotypingReader
from mimic3models import common_utils
from mimic3models import metrics
from mimic3models.phenotyping.utils import save_results

import numpy as np
import pandas as pd
import argparse
import os
import json

#only calculate mean for these columns
mean_only_columns = ['Capillary refill rate','Fraction inspired oxygen','Height']
#calculte full stats for these columns
calc_stat_columns = ['Diastolic blood pressure','Mean blood pressure','Oxygen saturation','Respiratory rate','Systolic blood pressure']
#stats to calculate
stats = ['min','max', 'mean', 'median', 'std', 'autocorr1', 'autocorr2', 'autocorr3', 'autocorr4', 'autocorr5' ]
#choose the endpoint
phenotype_header = ['Acute and unspecified renal failure', 'Acute cerebrovascular disease', 'Acute myocardial infarction', 'Cardiac dysrhythmias', 'Chronic kidney disease', 'Chronic obstructive pulmonary disease and bronchiectasis', 'Complications of surgical procedures or medical care', 'Conduction disorders', 'Congestive heart failure; nonhypertensive', 'Coronary atherosclerosis and other heart disease', 'Diabetes mellitus with complications', 'Diabetes mellitus without complication', 'Disorders of lipid metabolism', 'Essential hypertension', 'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage', 'Hypertension with complications and secondary hypertension', 'Other liver diseases', 'Other lower respiratory disease', 'Other upper respiratory disease', 'Pleurisy; pneumothorax; pulmonary collapse', 'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)', 'Respiratory failure; insufficiency; arrest (adult)', 'Septicemia (except in labor)', 'Shock']



def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    #ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_pennai_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'], ret['t'], ret['header'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--phenotype', type=str, default='Septicemia (except in labor)', help='specifies which endpoint to use for class',
                        choices=phenotype_header)
    parser.add_argument('--balance', dest='balanced', action='store_true')
    parser.set_defaults(balanced=False)
    parser.add_argument('--grid-search', dest='grid_search', action='store_true')
    parser.add_argument('--no-grid-search', dest='grid_search', action='store_false')
    parser.set_defaults(grid_search=False)
    parser.add_argument('--data', type=str, help='Path to the data of phenotyping task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/phenotyping/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()

    if args.grid_search:
        penalties = ['l2', 'l2', 'l2', 'l2', 'l2', 'l2', 'l1', 'l1', 'l1', 'l1', 'l1']
        coefs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1.0, 0.1, 0.01, 0.001, 0.0001]
    else:
        penalties = ['l1']
        coefs = [0.1]

    train_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                     listfile=os.path.join(args.data, 'train_listfile.csv'))

    val_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'train'),
                                   listfile=os.path.join(args.data, 'val_listfile.csv'))

    test_reader = PhenotypingReader(dataset_dir=os.path.join(args.data, 'test'),
                                    listfile=os.path.join(args.data, 'test_listfile.csv'))

    print('Reading data and extracting features ...')

    (train_X, train_y, train_names, train_ts, train_header) = read_and_extract_features(train_reader, args.period, args.features)
    train_y = np.array(train_y)

    (val_X, val_y, val_names, val_ts, val_header) = read_and_extract_features(val_reader, args.period, args.features)
    val_y = np.array(val_y)

    (test_X, test_y, test_names, test_ts, test_header) = read_and_extract_features(test_reader, args.period, args.features)
    test_y = np.array(test_y)

    summary_header = []
    if(train_header != test_header or train_header != val_header):
        print("something went wrong.  training and test headers do not match")
        exit()
    for i in train_header: 
        if(i != 'Hours'):
            for stat in stats:
                if(stat == 'mean' and i not in mean_only_columns or i in calc_stat_columns):
                    summary_header.append(i + '_' + stat) 
                else:
                    summary_header.append('deleteme_' + i + '_' + stat) 
    class_column = phenotype_header.index(args.phenotype)
    summary_header.append('class') 

    
    
    #
    print("train set shape:  {}".format(train_X.shape))
    print("validation set shape: {}".format(val_X.shape))
    print("test set shape: {}".format(test_X.shape))

    print('Imputing missing values ...')
    # impute for training
    train_imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
    train_imputer.fit(train_X)
    train_X = np.array(train_imputer.transform(train_X), dtype=np.float32)
    # impute for testing
    test_imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
    test_imputer.fit(test_X)
    test_X = np.array(test_imputer.transform(test_X), dtype=np.float32)
    # impute for validation
    val_imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
    val_imputer.fit(val_X)
    val_X = np.array(val_imputer.transform(val_X), dtype=np.float32)
    #
    train=np.append(train_X,train_y[:,class_column][:, None],axis=1)
    test=np.append(test_X,test_y[:,class_column][:, None],axis=1)
    val=np.append(val_X,val_y[:,class_column][:, None],axis=1)
    #
    train_summary=pd.DataFrame(data=train,columns=summary_header)
    train_summary.to_pickle(args.output_dir + "./train_summary.pkl")
    #
    test_summary=pd.DataFrame(data=test,columns=summary_header)
    test_summary.to_pickle(args.output_dir + "./test_summary.pkl")
    #
    val_summary=pd.DataFrame(data=val,columns=summary_header)
    val_summary.to_pickle(args.output_dir + "./val_summary.pkl")

    bal = ''
    if(args.balanced):
        bal='_bal'
        #create balanced training dataset
        class_count=min(train_summary.groupby('class').size())
        train_summary=train_summary.assign(class_count=0)
        count_true = 0
        count_false = 0
        for i, row in train_summary.iterrows():
            if (row['class'] == 1):
                train_summary.set_value(i,'class_count',count_true)
                count_true+=1
            if (row['class'] == 0):
                train_summary.set_value(i,'class_count',count_false)
                count_false+=1
        train_summary = train_summary[train_summary.class_count < class_count]
        train_summary = train_summary.drop(columns=['class_count'])
        #create balanced testing dataset
        class_count=min(train_summary.groupby('class').size())
        test_summary=test_summary.assign(class_count=0)
        count_true = 0
        count_false = 0
        for i, row in test_summary.iterrows():
            if (row['class'] == 1):
                test_summary.set_value(i,'class_count',count_true)
                count_true+=1
            if (row['class'] == 0):
                test_summary.set_value(i,'class_count',count_false)
                count_false+=1
        test_summary = test_summary[test_summary.class_count < class_count]
        test_summary = test_summary.drop(columns=['class_count'])
        #create balanced testing dataset
        class_count=min(val_summary.groupby('class').size())
        val_summary=val_summary.assign(class_count=0)
        count_true = 0
        count_false = 0
        for i, row in val_summary.iterrows():
            if (row['class'] == 1):
                val_summary.set_value(i,'class_count',count_true)
                count_true+=1
            if (row['class'] == 0):
                val_summary.set_value(i,'class_count',count_false)
                count_false+=1
        val_summary = test_summary[test_summary.class_count < class_count]



    #remove extra columns
    train_summary = train_summary[train_summary.columns.drop(list(train_summary.filter(regex='deleteme')))]
    test_summary = test_summary[test_summary.columns.drop(list(test_summary.filter(regex='deleteme')))]
    val_summary = val_summary[val_summary.columns.drop(list(val_summary.filter(regex='deleteme')))]
    #round values
    test_summary = test_summary.round(decimals=3)
    train_summary = train_summary.round(decimals=3)
    val_summary = val_summary.round(decimals=3)
    #save to csv
    train_summary.to_csv(args.output_dir + './train'+bal+'_summary.csv',index=False)
    test_summary.to_csv(args.output_dir + './test'+bal+'_summary.csv',index=False)
    val_summary.to_csv(args.output_dir + './val'+bal+'_summary.csv',index=False)
    print('done')

if __name__ == '__main__':
    main()
