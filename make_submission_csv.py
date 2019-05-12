import pandas as pd
def make_submission_nn(path1, path2, path3, submit_path):
    cat1=pd.read_pickle(path1)
    cat2=pd.read_pickle(path2)
    cat3=pd.read_pickle(path3)
    cat1[0].to_csv(submit_path, mode='w', index=False)
    for ii, df in enumerate(cat1):
        if ii==0: pass
        else:
            df.to_csv(submit_path, mode='a', index=False, header=False)
    for df in cat2:
        df.to_csv(submit_path, mode='a', index=False, header=False)
    for df in cat3:
        df.to_csv(submit_path, mode='a', index=False, header=False)
    df_check=pd.read_csv(submit_path)
    print(len(df_check))

    
make_submission_nn('./beauty_results/predictions_beauty.pickle', './fashion_results/predictions_fashion.pickle',
                  './mobile_results/predictions_mobile.pickle', './shopee_predictions10.csv')

