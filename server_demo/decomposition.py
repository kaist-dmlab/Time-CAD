import numpy as np
from statsmodels.tsa.seasonal import STL
from models import GRU_AE

def load_STL_results(test_df):
    train_trend, test_trend = [], []
    train_seasonal, test_seasonal = [], []
    train_resid, test_resid = [], []
    for i in range(test_df.shape[-1]):
        # stl = STL(train_df[:, i], seasonal=7, period=6*24*7)
        # results = stl.fit()
        # train_trend.append(results.trend.tolist())
        # train_seasonal.append(results.seasonal.tolist())
        # train_resid.append(results.resid.tolist())
        stl = STL(test_df[:, i], seasonal=7, period=6*24*7)
        results = stl.fit()
        test_trend.append(results.trend.tolist())
        test_seasonal.append(results.seasonal.tolist())        
        test_resid.append(results.resid.tolist())

    # train_trend = np.transpose(np.array(train_trend))
    test_trend = np.transpose(np.array(test_trend))
    
    # train_seasonal = np.transpose(np.array(train_seasonal))
    test_seasonal = np.transpose(np.array(test_seasonal))    

    # train_resid = np.transpose(np.array(train_resid))
    test_resid = np.transpose(np.array(test_resid))

    return {'test_trend': test_trend,'test_seasonal': test_seasonal, 'test_resid': test_resid}

def decompose_model(X_test):
    model = GRU_AE(X_test)
    # rec_train = model.predict(X_train)
    rec_test = model.predict(X_test)
    return {'rec_test': rec_test}
