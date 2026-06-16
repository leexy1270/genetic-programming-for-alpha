import alphalens as al
import pandas as pd
import numpy as np

load_data = np.load('data\\stock_data_3d.npz')
data = load_data["data_3d"]
stock_codes = load_data["stock_codes"].tolist()
trade_dates = pd.DatetimeIndex(load_data["dates"])
feature_cols = load_data["feature_cols"].tolist()

prices = pd.DataFrame(data[:,:,3].T,index=trade_dates,columns=stock_codes)

#load_result = np.load('result\\alpha_result.npz')




#help(al.utils.get_clean_factor_and_forward_returns)