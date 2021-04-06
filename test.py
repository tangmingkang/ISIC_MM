import pandas as pd

df_mm = pd.read_csv('mm.csv')
del df_mm['unknown']
df_train2 = pd.read_csv('train2.csv')
df_mm['label'] = df_mm['label'].fillna(-1)
df_train2['label'] = df_train2['label'].fillna(-1)
df_new = pd.merge( df_train2, df_mm, on = ['image'], how = "left" )
df_new['label_y'] = df_new['label_y'].fillna(-1)
def judgeLevel(df):
    if df['label_y'] < 0:
        return df['label_x']
    else:
        return df['label_y']
ax=0
df_new['label']=df_new.apply(lambda r:judgeLevel(r),axis=1)

def judgeLevel2(df):
    if df['label'] == 1.0 or df['label'] == 3.0 :
        return 1.0
    elif df['label'] == -1.0:
        return -1.0
    else:
        return 0.0
df_new['label']=df_new.apply(lambda r:judgeLevel2(r),axis=1)

del df_new['label_x']
del df_new['label_y']
for index, row in df_new.iterrows():
    if row['label']>=0:
        ax+=1
df_new=df_new.loc[df_new.label>=0]
df_new.to_csv('train_label.csv')
# print(df_new)
# df_newnew = df_new.loc[df_new.label>=0 & df_new.label<=3]
# df.dropna(subset=['label'])
# # print(df_train)
# print(df_train2)
# print(df_new)
# print(df_new)
# # print(df_newnew)
