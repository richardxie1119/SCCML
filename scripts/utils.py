import pandas as pd
from scipy.io import loadmat
import numpy as np



def loadmatfile(file_dir,norm_method,if_log):
    
    f = loadmat(file_dir)
    names = f['data']['names'][0][0]
    name_list = [str(i[0]) for i in names[0]]
    intens_matrix = f['data']['intens'][0][0].T
    mzs = f['data']['mzs'][0][0][0]
    mzs = np.around(mzs,4)

    print('Loaded intensity matrix with shape {}'.format(intens_matrix.shape))
    intens_df = pd.DataFrame(intens_matrix)
    intens_df = intens_df.set_index([pd.Index(name_list, 'ID')])
    #intens_df[intens_df==0]=1
    #intens_df = np.log(intens_df)
    intens_df.columns = mzs
    print(intens_df.shape)
    if norm_method == None:
        intens_df = intens_df
    if norm_method == 'l1':
        norm_factors = np.linalg.norm(intens_df,ord=1,axis=1)
        intens_df = intens_df/norm_factors.reshape(intens_df.shape[0],1)
    if norm_method == 'l2':
        norm_factors = np.linalg.norm(intens_df,ord=2,axis=1)
        intens_df = intens_df/norm_factors.reshape(intens_df.shape[0],1)
    if norm_method == 'max':
        norm_factors = intens_df.max(axis=1).values
        intens_df = intens_df/norm_factors.reshape(intens_df.shape[0],1)
    if norm_method =='mean':
        norm_factors = np.mean(intens_df.replace(0,np.NaN),axis=1).values
        norm_factors = norm_factors.reshape(intens_df.shape[0],1)
        intens_df = np.divide(intens_df,norm_factors)
    if norm_method == 'rms':
        norm_factors = np.sqrt(np.mean(intens_df.replace(0,np.NaN)**2,axis=1)).values
        norm_factors = norm_factors.reshape(intens_df.shape[0],1)
        intens_df = np.divide(intens_df,norm_factors)
    if norm_method == 'median':
        norm_factors = np.nanmedian(intens_df.replace(0,np.NaN),axis=1)
        intens_df = intens_df/norm_factors.reshape((intens_df.shape[0],1))
        
    if if_log == True:
        intens_df[intens_df==0] = 1
        intens_df = np.log2(intens_df)
        
    return intens_df,norm_factors

def loaddata(sample_names,batch_names,data_dir,norm_method,if_log):
    sample_dict = {}
    for batch_name in batch_names:
        batch_dict = {}
        for sample_name in sample_names:
            file_names = []
            for name in os.listdir('Data/{}/Slide1/'.format(batch_name)+sample_name):
                file_names.append(os.path.splitext(name)[0])
            batch_dict[sample_name] = file_names
        sample_dict[batch_name] = batch_dict
        
    data,n = loadmatfile(data_dir,norm_method,if_log)
    data_index = data.index
    for batch_name in sample_dict.keys():
        for sample_name in sample_dict[batch_name].keys():
            for file_name in sample_dict[batch_name][sample_name]:
                #data.loc[file_name, 'animal'] = sample_name[0]
                data.loc[file_name, 'type'] = sample_name[1]
                #data.loc[file_name, 'batch'] = batch_name
    return data

def split_data(data,test_size,random_state):
    

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size,random_state=random_state)

    data_dict = {'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}

    
    return data_dict