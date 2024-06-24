import numpy as np
import polars as pl
def datatype_compress(df):
    save_dict={}
    for column in df.columns:
        if (df[column].dtype!=pl.Int64) and (df[column].dtype!=pl.UInt32):continue
        column_min=df[column].min()
        column_max=df[column].max()
        if column_min>np.iinfo(np.int8).min and column_max<np.iinfo(np.int8).max:
            save_dict[column]=pl.Int8
        elif column_min>np.iinfo(np.int16).min and column_max<np.iinfo(np.int16).max:
            save_dict[column]=pl.Int16
        elif column_min>np.iinfo(np.int32).min and column_max<np.iinfo(np.int32).max:
            save_dict[column]=pl.Int32
        elif column_min>np.iinfo(np.int64).min and column_max<np.iinfo(np.int64).max:
            save_dict[column]=pl.Int64
    df=df.cast(save_dict)
    return df