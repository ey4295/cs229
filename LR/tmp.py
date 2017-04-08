""""
this is a tmp file for testing
"""

# Series
import numpy
import pandas
from pandas import Series, DataFrame

##1.construct
s1 = Series([4, 7, -5, 3])

##2.index&value
print (s1.values)
print (s1.index)
s1.index = ['a', 'b', 'c', 'd']
print (s1.index)
print (s1['a'])
print (s1 > 0)
print (s1[s1 > 0])
print (numpy.exp(s1))
print ('b' in s1)

##3.use dictionary to construct a Series
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
s2 = Series(sdata, index=['Ohio', 'Xuzhou', 'Texas'])
print (s2)

##4.auto alignment of different series
s3 = Series(sdata, index=['Ohio', 'Weinan', 'Utah'])
s4 = s2 + s3
print (s4)

##5.Series.name()
s1.name = 'city'
print (s1.name)

# dataframe
##1.construct
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
df = DataFrame(data)
print (df)

##2.columns
df = DataFrame(data, columns=['state', 'year', 'pop', 'area'])
print(df)
print (df['state'])

##3.index
df = DataFrame(data, columns=['state', 'year', 'pop', 'area'], index=['a', 'b', 'c', 'd', 'e'])
print (df)

##4.ix
print (df.ix['a'])
print (df.ix[0])

##5.set value for columns
df['area'] = 0
print (df)
df['area'] = numpy.arange(5.)  # 5. means 5.0
print(df)
s1 = Series([9, 45, 3], index=['c', 'd', 'e'])
df['area'] = s1
print (df)

##6.new & del column
df['pigs'] = 0
print(df)
del df['pigs']
print (df)

##7.nested dictionary construction
data = {'Xuzhou': {2001: 45, 2000: 23, 2005: 0}, 'Weinan': {2000: 1, 2001: 23}}
df = DataFrame(data)
print (df)

##8.transpose
print (df.T)

# applications
##1.reindex
data = [1, 2, 3, 4]
s1 = Series(data, index=['c', 'bull', 'd', 'f'])
print (s1)
s1 = s1.reindex(['a', 'b', 'c', 'd'], fill_value=0)
print(s1)

##2.reindex with fill(useful for time series analysis)
s2 = Series(['maomao', 'baobao', 'huahua'], index=[0, 3, 4])
print (s2)
s2 = s2.reindex(range(5), method='ffill')
print(s2)
s2 = s2.reindex(range(5), method='bfill')
print(s2)

##3.reindex for dataframe(rows)
df = DataFrame(numpy.arange(9).reshape([3, 3]), index=['a', 'b', 'c'], columns=['Ohio', 'Texas', 'California'])
print (df)
df = df.reindex(['a', 'b', 'c', 'd'])
print(df)

##4.reindex for dataframe(columns)
states = ['Texas', 'Ohio', 'California']
df = df.reindex(columns=states)
print (df)

##5.drop
df=df.drop(['a'])
print (df)

##6.ix
df=df.ix[['b','c'],['Texas','Ohio']]
print (df)

##7.dataframe & series
arr=numpy.arange(12).reshape([3,4])
s1=df.ix['b']
print (s1)
df=df-s1
print (df)


##8.pandas read txt
#t=pandas.read_table('logistic_x.txt',header=None,dtype=float)

X=pandas.read_table('logistic_x.txt',sep=' +',header=None)
Y=pandas.read_table('logistic_y.txt',sep=' +',header=None)
X.columns=['x1','x2']
print (X)
print (Y)

print ('here is the result')
print (X.shape)