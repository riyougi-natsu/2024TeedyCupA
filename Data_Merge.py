from Data_Compress import datatype_compress
import polars as pl
def loading_data():
    needed_columns = ['日期', '时间', '生产线编号', '物料推送气缸推送状态', '物料推送气缸收回状态', '物料推送数',
                      '物料待抓取数', '容器上传检测数', '填装检测数', '填装定位器固定状态', '填装定位器放开状态',
                      '物料抓取数', '填装旋转数', '填装下降数', '加盖检测数', '加盖定位数', '推盖数', '加盖下降数',
                      '拧盖检测数', '拧盖定位数', '拧盖下降数', '拧盖旋转数', '拧盖数', '合格数', '不合格数',
                      '物料推送装置故障1001', '物料检测装置故障2001', '填装装置检测故障4001', '填装装置定位故障4002',
                      '填装装置填装故障4003', '加盖装置定位故障5001', '加盖装置加盖故障5002', '拧盖装置定位故障6001',
                      '拧盖装置拧盖故障6002']
    train_df = datatype_compress(pl.read_csv(f'..\\dataset\\M101.csv', columns=needed_columns))
    read_nums = 10
    for i in range(1, read_nums):
        if i == 9:
            temp_df = datatype_compress(pl.read_csv(f'..\\dataset\\M110.csv', columns=needed_columns))
        else:
            temp_df = datatype_compress(pl.read_csv(f'D..\\dataset\\M10{i + 1}.csv', columns=needed_columns))
        train_df = pl.concat([train_df, temp_df])
    return train_df