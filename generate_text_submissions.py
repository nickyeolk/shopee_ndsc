import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

def create_submission_list(trainpath, validpath):
    df_train=pd.read_csv(trainpath)
#     df_train=df_train.fillna(-1)
#     df_train=treatna2(df_train)
    items=[thing for thing in df_train.columns if thing not in ['itemid', 'image_path', 'title']]
    print(items)
    vectorizer = CountVectorizer(analyzer = "word", strip_accents=None, tokenizer = None, preprocessor = None, \
                                 stop_words = None, max_features = 10000, ngram_range=(1,3))
    train_data_features = vectorizer.fit_transform(df_train['title'])
    tfidfier = TfidfTransformer()
    train_tfidf = tfidfier.fit_transform(train_data_features)
    df_valid=pd.read_csv(validpath)
    valid_data_features=vectorizer.transform(df_valid['title'])
    valid_tfidf=tfidfier.transform(valid_data_features)
    ids=df_valid['itemid']
    predictions=[]
    for item in items:
        print('training for ', item)
        lr=LogisticRegression(solver='lbfgs', multi_class='ovr', n_jobs=-1)
#         lr=GaussianNB()
        nanmask=df_train.notna()[item].values
        df_item=df_train.dropna(axis=0, subset=[item])
        lr.fit(train_tfidf[nanmask], df_item[item])
        print('predicting for ', item)
        allprobs=lr.predict_proba(valid_tfidf)
        sortprobs=np.argsort(allprobs, axis=1)[:, ::-1]
        classes=lr.classes_
        y_predicts=classes[sortprobs]
        ids_feature=[str(x)+'_'+item for x in ids]
        tags=[]
        for item in y_predicts:
            tags.append(" ".join(map(str, map(int, item))))
#         tags = [w.replace('-1 ', '') for w in tags]
        predictions.append(pd.DataFrame({'id':ids_feature, 'tagging':tags}))
    return predictions

if __name__ == '__main__':
    predictions_beauty=create_submission_list('./data/beauty_data_info_train_competition.csv', \
                                        './data/beauty_data_info_val_competition.csv')
    predictions_fashion=create_submission_list('./data/fashion_data_info_train_competition.csv', \
                                           './data/fashion_data_info_val_competition.csv')
    predictions_mobile=create_submission_list('./data/mobile_data_info_train_competition.csv', \
                                           './data/mobile_data_info_val_competition.csv')
    all_len=0
    for df in predictions_beauty:
        all_len+=len(df)
    for df in predictions_fashion:
        all_len+=len(df)
    for df in predictions_mobile:
        all_len+=len(df)
    print(all_len)    

    submit_path='./shopee_predictions11.csv'
    predictions_beauty[0].to_csv(submit_path, mode='w', index=False)
    for ii, df in enumerate(predictions_beauty):
        if ii==0: pass
        else:
            df.to_csv(submit_path, mode='a', index=False, header=False)
    for df in predictions_fashion:
        df.to_csv(submit_path, mode='a', index=False, header=False)
    for df in predictions_mobile:
        df.to_csv(submit_path, mode='a', index=False, header=False)
    print('finish3!')