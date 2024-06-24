import lightgbm as lgb
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import f1_score, log_loss
def cv_models(data,x_test):
    test_results=[] # 存储每次交叉验证的模型所预测的结果
    params = { # 设置参数（有待修改）
        'learning_rate': 0.01,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': -1,
        'objective': 'multiclass',  # 目标函数
        'num_class': 10,
    }
    folds = KFold(n_splits=5, shuffle=True, random_state=2024)
    f1_scores=[]
    losses=[]
    for i, (trn_index, val_index) in enumerate(folds.split(data)):
        print('the %dth fold' % (i + 1))
        # 划分数据集
        trn_x = data.iloc[trn_index].drop('target',axis=1)
        trn_y = data.iloc[trn_index]['target']
        val_x = data.iloc[val_index].drop('target',axis=1)
        val_y = data.iloc[val_index]['target']
        # 将数据转化为lgbm训练专用格式
        trn_data = lgb.Dataset(trn_x, label=trn_y)
        val_data = lgb.Dataset(val_x, label=val_y)
        # 模型训练
        model=lgb.train(
            params=params,
            train_set=trn_data,
            valid_sets=[trn_data,val_data],
        )
        # 模型预测
        y_pred_prob = model.predict(val_x)
        y_pred = np.argmax(y_pred_prob,axis=1)
        test_results.append(np.argmax(model.predict(x_test.drop('target',axis=1)),axis=1))
        # 评价指标
        f1=f1_score(y_pred=y_pred,y_true=val_y,average='weighted')
        f1_scores.append(f1)
        loss = log_loss(y_true=val_y, y_pred=y_pred_prob)
        losses.append(loss)
        print(f"f1 score:{f1}")
        print(f"loss:{loss}")
    print(np.mean(f1_scores))
    print(np.mean(losses))
    return test_results