## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![321847414-aae763cd-3117-4c8e-bf2c-c652a5541541](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/bd3cab7e-1d40-43e5-bedc-15a5bfe99221)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![321847451-b8e416e0-1631-4afc-a303-fba2a3a1febd](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/41c95ca4-d692-4638-a71c-2c9c0eef7025)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![321847486-2cbc1dc1-5234-45a6-a925-840d7e0baee2](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/7c8ae7ba-5a65-4d9d-99a8-1b032f8c74f3)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![321847550-12ce4365-d23e-4986-941c-b26b61a6bdd7](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/8cd5e10b-46b7-4ff0-bef3-3fbd8b0c44c7)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![321847630-d6b8bdc9-91e6-41d6-b852-e7c343cc096a](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/39054d3e-80de-4282-a9a3-952abc21e5da)


```
df2=pd.concat([df2,enc],axis=1)
df2
```
![321847663-df24a068-454c-426e-b5c6-77eeb69e719b](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/ccd04184-6234-44c3-8b7e-ec6feb9d0675)


```
pd.get_dummies(df2,columns=["nom_0"])
```
![321847704-54fdc26f-c5e7-4026-a48d-707aee14faab](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/0b0552e2-637a-42bd-858f-2cd077d53dd6)


```
pip install --upgrade category_encoders
```
![321847769-9deca6ac-1360-4800-93df-d673c5ccd415](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/a5379f7d-07c1-49ee-85b6-a28cb28c5e6d)


```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![321847807-4c46dc34-429c-489c-bf3a-ce2c62d0fe24](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/645243d0-4bc1-4c4a-9759-9d86f3726c61)


```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![321847853-7e5f8d02-39f8-429b-a1a6-8223f63cd417](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/bbe68b52-55b5-4f97-b829-a7c19520fdfb)

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![321847889-24b9257d-f1a0-4284-9cfe-bcebe6f13261](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/18b92f18-a286-46b1-b5da-c483fbd564f6)


```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![321847991-a7944f5e-dac7-41e6-a061-f1326ce96f1c](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/e2061e4b-dbcc-4cfb-a375-25481e588fed)


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![321848026-29e2129f-10e1-448e-9741-1580cbf49b64](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/c3a8ee58-ba39-43ef-a71d-b73898b3521c)


```
df.skew()
```
![321848068-bdc4afd2-b0f2-4306-b4c2-66b1b369c7f7](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/87162272-b2dc-40a7-8528-f6a6f9791b73)


```
np.log(df["Highly Positive Skew"])
```
![321848220-b78a4f39-f466-46d9-a565-f0e8a8e1a891](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/6943820b-42bc-46e5-83a5-12b5dfb86412)


```
np.reciprocal(df["Moderate Positive Skew"])
```
![321848260-2b4156b5-dca8-4763-b2d1-384242df4b23](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/42b3c689-1189-4c3b-a9c6-a90ff835969a)


```
np.sqrt(df["Highly Positive Skew"])
```
![321848296-f1d49252-4997-4dc3-b905-438a91af37a8](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/271d1fcc-6b69-47fc-b4db-bfa51004631e)


```
np.square(df["Highly Positive Skew"])
```
![321848336-01c8afd9-3287-4c58-977f-c43db1aba7ff](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/4cbeb569-c4bc-4328-900e-172ab9e2bb4c)


```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![321848376-1f345c86-c0f1-4a7c-99c7-9d7f7ee24ce8](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/a9e5b2ae-19d4-4322-8e44-05a1b38759dc)


```
df.skew()
```
![321848404-a1fcc12d-2f68-4951-b8ba-ce5a13544d23](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/2f55ff8a-7e16-4330-9b63-7c8beb5ee6f5)


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```
![321848440-d84c61ec-dd51-45ae-a2d1-0b7272d8e630](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/f37265ff-f23d-4b5b-950a-65a34f4f5d68)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![321848463-09a83e2b-950f-463f-9e97-bea2426780be](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/86c52c21-7a80-4cd8-b95d-b7ee4ab96c85)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![321848531-9f8a1915-eeba-4959-a64f-2e469c1a6894](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/aa779a99-8589-43d4-82a0-55e0c6c2a535)


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![321848576-e3338bb1-b492-4cd5-b32b-02e2430bc9da](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/b781cee6-ecde-49db-ad3e-3ca07f2571f8)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![321848611-b732faef-2b5a-4265-a643-f86e382bbc04](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/2ed694f2-0a49-44e1-9c57-2a0af2fd9c84)


```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![321848673-3f12d222-7685-4aa9-8288-c1717a69c014](https://github.com/dharani18p/EXNO-3-DS/assets/118343366/a2157fd1-ceed-4555-bffe-ac9f85bba0a0)



# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
