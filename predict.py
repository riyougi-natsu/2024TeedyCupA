from Feature_Engineering import pre_treat
import polars as pl
import lightgbm as gbm
import numpy as np
from Data_Merge import datatype_compress
test1=pre_treat(datatype_compress(pl.read_csv(".\\testdata\\M201.csv"))).drop(['date','time','line_id','target'])
test2=pre_treat(datatype_compress(pl.read_csv(".\\testdata\\M201.csv"))).drop(['date','time','line_id','target'])
model=gbm.Booster(model_file=".\\lightgbm_model.txt")

pred1=np.argmax(model.predict(test1),axis=1)
pred2=np.argmax(model.predict(test2),axis=1)

def result_fix(df,pred):
    result_analysis = df[['date', 'time', 'target']]
    result_analysis = result_analysis.with_columns(pl.Series(pred).alias('pred'))
    result_analysis = result_analysis.with_columns(
        (pl.col('pred') != pl.col('pred').shift()).cum_sum().fill_null(0).alias('pred_group')
    )
    result_analysis = result_analysis.with_columns(
        pl.col('pred').count().over('pred_group').alias('pred_count')
    )
    result_analysis = result_analysis.with_columns(
        pl.when(((pl.col('pred') == 0) & (pl.col('pred_count') <= 40)) | (pl.col('pred') != 0))
        .then(1).otherwise(0).alias('alarm_needed')
    )
    result_analysis = result_analysis.with_columns(
        (pl.col('alarm_needed') != pl.col('alarm_needed').shift()).cum_sum().fill_null(0).alias('alarm_group')
    )
    result_analysis = datatype_compress(result_analysis)
    result_analysis = result_analysis.join(
        datatype_compress(result_analysis.group_by('alarm_group').agg(pl.col('pred').mode().alias('temp_pred')).sort(
            'alarm_group').map_rows(lambda x: (x[0], x[1][0])).rename(
            {'column_0': 'alarm_group', 'column_1': 'temp_pred'})),
        on='alarm_group'
    )
    result_analysis = result_analysis.with_columns(
        (pl.col('temp_pred') != pl.col('temp_pred').shift()).cum_sum().fill_null(0).alias('temp_pred_group')
    )
    result_analysis = result_analysis.with_columns(
        pl.col('temp_pred').count().over('temp_pred_group').alias('temp_pred_count')
    )
    result_analysis = result_analysis.with_columns(
        pl.when((pl.col('temp_pred') != 0) & (pl.col('temp_pred_count') <= 20))
        .then(0).otherwise(pl.col('temp_pred')).alias('fixed_pred')
    )
    pred_result = result_analysis[['date', 'time', 'target', 'fixed_pred']].rename({'fixed_pred': 'pred'})
    return pred_result

def submit(df):
    submit_result = df.filter(pl.col('pred') != 0).group_by('pred_group') \
        .agg(
        pl.col('date').first(),
        pl.col('time').first(),
        pl.col('pred').first(),
        pl.col('pred_count').first()
    ).drop('pred_group').sort('date')
    result = submit_result.filter(pl.col('pred') == 1).drop('pred') \
        .rename({'date': '日期_1', 'time': '开始时间_1', 'pred_count': '持续时长/秒_1'}).to_pandas()
    for i in range(2, 10):
        result = result.join(submit_result.filter(pl.col('pred') == i).drop('pred') \
                             .rename(
            {'date': f'日期{i}', 'time': f'开始时间_{i}', 'pred_count': f'持续时长/秒_{i}'}).to_pandas())
    return result

result1=submit(result_fix(test1,pred1))
result2=submit(result_fix(test2,pred2))