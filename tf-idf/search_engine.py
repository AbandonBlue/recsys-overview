"""
    目的:
        透過搜尋輸入用TFIDF轉換為向量，與資料庫的文本作相似性比對，返回最相關的內容。

    步驟:
        1. 取得資料庫所有文章之內容 + 搜尋token
        2. 將上面之內容選取文本內容，透過TFIDF轉換
        3. 透過df排序，透過相似度排序。
        4. 最後返回ids給主程式使用。
"""
import sqlite3
import numpy as np
import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances


def get_all_data(sqlite_file: str='data.sqlite', sql_cmd: str='select * from log'):
    """ 取得資料庫資料 """
    with sqlite3.connect(sqlite_file) as con:
        df = pd.read_sql(sql_cmd, con=con)
    return df


def transform_data(df: pd.DataFrame, search_token):
    """ 透過get_all_data得到之資料做清理、轉換，得到乾淨之df、corpus """
    # 文本處理
    df['title'] = df['title'].apply(func=lambda x: x.lower())
    df['content'] = df['content'].apply(func=lambda x: x.lower())
    df['content'] = df['title'] + df['content']

    # 斷詞
    jieba.load_userdict('new_dict.txt')
    word_vectors = [list(jieba.cut(search_token))]           # 之後會做成corpus, 目前是list of list of word
    for i in range(len(df)):
        word_vectors.append(list(jieba.cut(df.iloc[i, 2])))     # 因為content在index 2
    
    # 去除停用詞
    stopwords_eng = set(stopwords.words('english'))
    with open(r'C:\Users\aband\OneDrive\桌面\NLP_marathon\NLP_practice\1-st_NLP\hw\datasets\停用詞-繁體中文.txt', 'r', encoding='utf8') as f:
        stopwords_chinese = set(f.read().split('\n'))
    stopwords_all = stopwords_chinese.union(stopwords_eng)
    for i in range(len(word_vectors)):
        for j in range(len(word_vectors[i])-1, -1, -1):
            if word_vectors[i][j] in stopwords_all:
                word_vectors[i].pop(j)
    df = pd.concat([pd.DataFrame([(-1, '', search_token)], columns=df.columns), df])     # 給於search token一個位置
    return df, word_vectors


def tfidf(word_vectors: list, df: pd.DataFrame):
    """ 將word_vectors轉換成corpus進行tfidf計算 """
    corpus = []
    for i in range(len(word_vectors)):
        corpus.append(' '.join(word_vectors[i]))
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)  # 此時為稀疏矩陣
    # 此時將id與td-idf向量做合併, 但因為search token沒有id, 為了方面, 給他-1 
    df_ids = pd.DataFrame(df.iloc[:, 0], columns=['id_'])        # 避開: "Reindexing only valid with uniquely valued Index objects" pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued Index objects
    df_vectors = X.toarray()
    df = np.concatenate([df_ids, df_vectors], axis=1)
    df = pd.DataFrame(df, columns=['id_'] + [i for i in range(df.shape[1]-1)])
    return df

def get_notes_df_idf(df: pd.DataFrame, id_: int=-1):
    # 把id_放在第一個row
    df_sorted = df.copy()
    df_sorted = pd.concat([df_sorted[df_sorted['id_'] == id_], df_sorted[df_sorted['id_'] != id_]])
    
    df_features = df_sorted.iloc[:, 1:]   # 因為沒用到上面的!所以方便之後擴充先改名，這一行取得所有df的特徵。
    
    # 計算距離
    X = df_features.values
    Y = df_features.values[0].reshape(1, -1)    # 第一筆, 也就是目前看的notes
    distances = cosine_similarity(X, Y)                    # 可以換
    
    df_sorted['similarity_distance'] = distances
    return df_sorted.sort_values('similarity_distance', ascending=False).reset_index(drop=True)


def get_search_result(search_token: str='搜尋', k: int=20):
    df = get_all_data('data.sqlite', 'select id_, title, content from  log')
    # input(df)
    df, word_vectors = transform_data(df, search_token)
    # input(df)
    # input(word_vectors)
    df_new = tfidf(word_vectors, df)
    # input(df_new)
    df_cosine = get_notes_df_idf(df_new)
    # input(df_cosine)
    # print(df_cosine.shape)
    print(list(df_cosine.iloc[1:k+1, 0]))
    return list(df_cosine.iloc[1:k+1, 0])


if __name__ == '__main__':
    print(get_search_result('推薦系統、資料科學', 10))