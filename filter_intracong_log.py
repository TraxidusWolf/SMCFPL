import pandas as pd

df = pd.read_csv('IntraCongs.log', header=None, usecols=[2,3,4,5,6], names=['StageNum', 'CaseNum', 'TypeElmnt', 'IndxTable', 'loading_percent'])
Tbl = df.groupby(by=['StageNum', 'TypeElmnt', 'IndxTable']).describe()
print(Tbl)
