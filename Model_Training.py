from Data_Merge import loading_data
from Feature_Engineering import pre_treat
import Feature_Engineering
import lightgbm as lgb
train_df=pre_treat(loading_data()).drop(['date','time','line_id'])
train=lgb.Dataset(train_df.drop('target'),label=train_df['target']).to_pandas()
params = {
    'learning_rate': 0.01,
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    'max_depth': -1,
    'objective': 'multiclass',  # 目标函数
    'num_class': 10,
}
gbm=lgb.train(params,train)
gbm.save_model(".\\lightgbm_model.txt")