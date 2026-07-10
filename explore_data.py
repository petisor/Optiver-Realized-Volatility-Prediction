import pandas as pd
train=pd.read_csv("data/train.csv")
print(train.shape)
print(train.columns)
print(train.head())

book=pd.read_csv("data/book_train.parquet/stock_id=0")
print(book.shape)
print(book.columns)
print(book.head())