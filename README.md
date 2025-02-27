# 导入文件


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```


```python
data = pd.read_csv('amazon.csv')
```

# 一、数据预处理：替换缺失值为0


```python
for col in ['discounted_price',	'actual_price',	'discount_percentage','rating',	'rating_count']:
  data[col]=data[col].astype(str).str.replace('₹','').str.replace('%','').str.replace(',','').str.replace('nan','0')
  data[col] = pd.to_numeric(data[col].str.replace('[^0-9.]','',regex=True),errors='coerce').fillna(0)
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>category</th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>about_product</th>
      <th>user_id</th>
      <th>user_name</th>
      <th>review_id</th>
      <th>review_title</th>
      <th>review_content</th>
      <th>img_link</th>
      <th>product_link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B07JW9H4J1</td>
      <td>Wayona Nylon Braided USB to Lightning Fast Cha...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>399.0</td>
      <td>1099.0</td>
      <td>64</td>
      <td>4.2</td>
      <td>24269</td>
      <td>High Compatibility : Compatible With iPhone 12...</td>
      <td>AG3D6O4STAQKAY2UVGEUV46KN35Q,AHMY5CWJMMK5BJRBB...</td>
      <td>Manav,Adarsh gupta,Sundeep,S.Sayeed Ahmed,jasp...</td>
      <td>R3HXWT0LRP0NMF,R2AJM3LFTLZHFO,R6AQJGUP6P86,R1K...</td>
      <td>Satisfied,Charging is really fast,Value for mo...</td>
      <td>Looks durable Charging is fine tooNo complains...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Wayona-Braided-WN3LG1-Sy...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B098NS6PVG</td>
      <td>Ambrane Unbreakable 60W / 3A Fast Charging 1.5...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>199.0</td>
      <td>349.0</td>
      <td>43</td>
      <td>4.0</td>
      <td>43994</td>
      <td>Compatible with all Type C enabled devices, be...</td>
      <td>AECPFYFQVRUWC3KGNLJIOREFP5LQ,AGYYVPDD7YG7FYNBX...</td>
      <td>ArdKn,Nirbhay kumar,Sagar Viswanathan,Asp,Plac...</td>
      <td>RGIQEG07R9HS2,R1SMWZQ86XIN8U,R2J3Y1WL29GWDE,RY...</td>
      <td>A Good Braided Cable for Your Type C Device,Go...</td>
      <td>I ordered this cable to connect my phone to An...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Ambrane-Unbreakable-Char...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B096MSW6CT</td>
      <td>Sounce Fast Phone Charging Cable &amp; Data Sync U...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>199.0</td>
      <td>1899.0</td>
      <td>90</td>
      <td>3.9</td>
      <td>7928</td>
      <td>【 Fast Charger&amp; Data Sync】-With built-in safet...</td>
      <td>AGU3BBQ2V2DDAMOAKGFAWDDQ6QHA,AESFLDV2PT363T2AQ...</td>
      <td>Kunal,Himanshu,viswanath,sai niharka,saqib mal...</td>
      <td>R3J3EQQ9TZI5ZJ,R3E7WBGK7ID0KV,RWU79XKQ6I1QF,R2...</td>
      <td>Good speed for earlier versions,Good Product,W...</td>
      <td>Not quite durable and sturdy,https://m.media-a...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Sounce-iPhone-Charging-C...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B08HDJ86NZ</td>
      <td>boAt Deuce USB 300 2 in 1 Type-C &amp; Micro USB S...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>329.0</td>
      <td>699.0</td>
      <td>53</td>
      <td>4.2</td>
      <td>94363</td>
      <td>The boAt Deuce USB 300 2 in 1 cable is compati...</td>
      <td>AEWAZDZZJLQUYVOVGBEUKSLXHQ5A,AG5HTSFRRE6NL3M5S...</td>
      <td>Omkar dhale,JD,HEMALATHA,Ajwadh a.,amar singh ...</td>
      <td>R3EEUZKKK9J36I,R3HJVYCLYOY554,REDECAZ7AMPQC,R1...</td>
      <td>Good product,Good one,Nice,Really nice product...</td>
      <td>Good product,long wire,Charges good,Nice,I bou...</td>
      <td>https://m.media-amazon.com/images/I/41V5FtEWPk...</td>
      <td>https://www.amazon.in/Deuce-300-Resistant-Tang...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B08CF3B7N1</td>
      <td>Portronics Konnect L 1.2M Fast Charging 3A 8 P...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>154.0</td>
      <td>399.0</td>
      <td>61</td>
      <td>4.2</td>
      <td>16905</td>
      <td>[CHARGE &amp; SYNC FUNCTION]- This cable comes wit...</td>
      <td>AE3Q6KSUK5P75D5HFYHCRAOLODSA,AFUGIFH5ZAFXRDSZH...</td>
      <td>rahuls6099,Swasat Borah,Ajay Wadke,Pranali,RVK...</td>
      <td>R1BP4L2HH9TFUP,R16PVJEXKV6QZS,R2UPDB81N66T4P,R...</td>
      <td>As good as original,Decent,Good one for second...</td>
      <td>Bought this instead of original apple, does th...</td>
      <td>https://m.media-amazon.com/images/W/WEBP_40237...</td>
      <td>https://www.amazon.in/Portronics-Konnect-POR-1...</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1465 entries, 0 to 1464
    Data columns (total 16 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   product_id           1465 non-null   object 
     1   product_name         1465 non-null   object 
     2   category             1465 non-null   object 
     3   discounted_price     1465 non-null   float64
     4   actual_price         1465 non-null   float64
     5   discount_percentage  1465 non-null   int64  
     6   rating               1465 non-null   float64
     7   rating_count         1465 non-null   int64  
     8   about_product        1465 non-null   object 
     9   user_id              1465 non-null   object 
     10  user_name            1465 non-null   object 
     11  review_id            1465 non-null   object 
     12  review_title         1465 non-null   object 
     13  review_content       1465 non-null   object 
     14  img_link             1465 non-null   object 
     15  product_link         1465 non-null   object 
    dtypes: float64(3), int64(2), object(11)
    memory usage: 183.2+ KB



```python
data.columns
```




    Index(['product_id', 'product_name', 'category', 'discounted_price',
           'actual_price', 'discount_percentage', 'rating', 'rating_count',
           'about_product', 'user_id', 'user_name', 'review_id', 'review_title',
           'review_content', 'img_link', 'product_link'],
          dtype='object')




```python
data.isnull().sum()
```




    product_id             0
    product_name           0
    category               0
    discounted_price       0
    actual_price           0
    discount_percentage    0
    rating                 0
    rating_count           0
    about_product          0
    user_id                0
    user_name              0
    review_id              0
    review_title           0
    review_content         0
    img_link               0
    product_link           0
    dtype: int64



# 二、EDA 描述性统计

# EDA：对数值列进行描述性统计分析，罗列出最大值，最小值，标准差; 25%, 50%, 75%置信区间


```python
data[['discounted_price','actual_price','discount_percentage']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1465.000000</td>
      <td>1465.000000</td>
      <td>1465.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3125.310874</td>
      <td>5444.990635</td>
      <td>47.691468</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6944.304394</td>
      <td>10874.826864</td>
      <td>21.635905</td>
    </tr>
    <tr>
      <th>min</th>
      <td>39.000000</td>
      <td>39.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>325.000000</td>
      <td>800.000000</td>
      <td>32.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>799.000000</td>
      <td>1650.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1999.000000</td>
      <td>4295.000000</td>
      <td>63.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77990.000000</td>
      <td>139900.000000</td>
      <td>94.000000</td>
    </tr>
  </tbody>
</table>
</div>



# 显示不同级别的评分共出现了多少次


```python
data['rating'].value_counts()
```




    4.1    244
    4.3    230
    4.2    228
    4.0    181
    3.9    123
    4.4    123
    3.8     86
    4.5     75
    3.7     42
    3.6     35
    3.5     26
    4.6     17
    3.3     16
    3.4     10
    4.7      6
    3.0      4
    3.1      4
    5.0      3
    4.8      3
    3.2      2
    2.8      2
    2.3      1
    0.0      1
    2.0      1
    2.6      1
    2.9      1
    Name: rating, dtype: int64



# 按照类别分组，并且给予平均值降序排列


```python
data.groupby('category')['rating'].mean().sort_values(ascending=False)
```




    category
    Computers&Accessories|Tablets                                                                                    4.60
    Computers&Accessories|NetworkingDevices|NetworkAdapters|PowerLANAdapters                                         4.50
    Electronics|Cameras&Photography|Accessories|Film                                                                 4.50
    Electronics|HomeAudio|MediaStreamingDevices|StreamingClients                                                     4.50
    OfficeProducts|OfficeElectronics|Calculators|Basic                                                               4.50
                                                                                                                     ... 
    Electronics|HomeTheater,TV&Video|Accessories|3DGlasses                                                           3.50
    Computers&Accessories|Accessories&Peripherals|Audio&VideoAccessories|PCHeadsets                                  3.50
    Home&Kitchen|Kitchen&HomeAppliances|Vacuum,Cleaning&Ironing|Vacuums&FloorCare|Vacuums|HandheldVacuums            3.45
    Computers&Accessories|Accessories&Peripherals|Keyboards,Mice&InputDevices|Keyboard&MiceAccessories|DustCovers    3.40
    Home&Kitchen|Kitchen&HomeAppliances|Coffee,Tea&Espresso|CoffeeGrinders|ElectricGrinders                          3.30
    Name: rating, Length: 211, dtype: float64



# EDA: 对数值列进行初步可视化


```python
data.hist(bins=50, figsize=(20, 15))
```




    array([[<AxesSubplot:title={'center':'discounted_price'}>,
            <AxesSubplot:title={'center':'actual_price'}>],
           [<AxesSubplot:title={'center':'discount_percentage'}>,
            <AxesSubplot:title={'center':'rating'}>],
           [<AxesSubplot:title={'center':'rating_count'}>, <AxesSubplot:>]],
          dtype=object)




    
![png](output_17_1.png)
    


# EDA: 对4列数据进行相关性分析


```python
from pandas.plotting import scatter_matrix
attributes = ['discounted_price', 'actual_price', 'discount_percentage', 'rating']
scatter_matrix(data[attributes], figsize=(12, 8))
```




    array([[<AxesSubplot:xlabel='discounted_price', ylabel='discounted_price'>,
            <AxesSubplot:xlabel='actual_price', ylabel='discounted_price'>,
            <AxesSubplot:xlabel='discount_percentage', ylabel='discounted_price'>,
            <AxesSubplot:xlabel='rating', ylabel='discounted_price'>],
           [<AxesSubplot:xlabel='discounted_price', ylabel='actual_price'>,
            <AxesSubplot:xlabel='actual_price', ylabel='actual_price'>,
            <AxesSubplot:xlabel='discount_percentage', ylabel='actual_price'>,
            <AxesSubplot:xlabel='rating', ylabel='actual_price'>],
           [<AxesSubplot:xlabel='discounted_price', ylabel='discount_percentage'>,
            <AxesSubplot:xlabel='actual_price', ylabel='discount_percentage'>,
            <AxesSubplot:xlabel='discount_percentage', ylabel='discount_percentage'>,
            <AxesSubplot:xlabel='rating', ylabel='discount_percentage'>],
           [<AxesSubplot:xlabel='discounted_price', ylabel='rating'>,
            <AxesSubplot:xlabel='actual_price', ylabel='rating'>,
            <AxesSubplot:xlabel='discount_percentage', ylabel='rating'>,
            <AxesSubplot:xlabel='rating', ylabel='rating'>]], dtype=object)




    
![png](output_19_1.png)
    


#  我们着重关注最后一行的用户最终评分，可以看到评分与 打折后的价格、原价格、打折力度有着明显的线性相关趋势。尤其关注打折力度和最终评分的关系，之后可以考虑用线性回归模型进行建模分析，下面我们查看他们之间的皮尔逊相关系数


```python
corr_matrix = data[['discounted_price','actual_price','discount_percentage', 'rating']].corr()
```


```python
corr_matrix
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>discounted_price</th>
      <td>1.000000</td>
      <td>0.961915</td>
      <td>-0.242412</td>
      <td>0.114298</td>
    </tr>
    <tr>
      <th>actual_price</th>
      <td>0.961915</td>
      <td>1.000000</td>
      <td>-0.118098</td>
      <td>0.116629</td>
    </tr>
    <tr>
      <th>discount_percentage</th>
      <td>-0.242412</td>
      <td>-0.118098</td>
      <td>1.000000</td>
      <td>-0.132556</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>0.114298</td>
      <td>0.116629</td>
      <td>-0.132556</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
```


    
![png](output_23_0.png)
    


# 可以看到两者之间的皮尔逊系数为-0.13，打折力度增加用户评分反而可能下降，这一现象可能符合买家的某些心理，例如：最贵的一定是最好的

# EDA: 深入探索，这次区分商品类别


```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="discount_percentage", y="rating", hue="category", alpha=0.7)
plt.title("Discount Percentage vs. Rating")
plt.xlabel("Discount Percentage")
plt.ylabel("Rating")
plt.legend(title="Category", bbox_to_anchor=(1, 1))
plt.show()
```


    
![png](output_26_0.png)
    


# EDA:使用wordcloud统计用户评论中的高频词汇


```python
from wordcloud import WordCloud

text = " ".join(data['review_content'].dropna())  # Combine all reviews
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Common Words in Product Reviews")
plt.show()


```


    
![png](output_28_0.png)
    


# 三、使用决策树模型建模

### 对类别数据进行独热编码


```python
data['categories'] = data['category'].str.split('|')
```


```python
data['categories'].head()
```




    0    [Computers&Accessories, Accessories&Peripheral...
    1    [Computers&Accessories, Accessories&Peripheral...
    2    [Computers&Accessories, Accessories&Peripheral...
    3    [Computers&Accessories, Accessories&Peripheral...
    4    [Computers&Accessories, Accessories&Peripheral...
    Name: categories, dtype: object




```python
from sklearn.preprocessing import OneHotEncoder
data['categories'] = data['categories'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
cat_encoder = OneHotEncoder()
data_cat_1hot = cat_encoder.fit_transform(data[['categories']])
```


```python
data_cat_1hot
```




    <1465x211 sparse matrix of type '<class 'numpy.float64'>'
    	with 1465 stored elements in Compressed Sparse Row format>




```python
# Convert the one-hot encoded array to a DataFrame
encoded_df = pd.DataFrame(data_cat_1hot.toarray(), columns=cat_encoder.categories_[0])

# Concatenate the original DataFrame with the one-hot encoded DataFrame
data = pd.concat([data, encoded_df], axis=1)
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>category</th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>about_product</th>
      <th>user_id</th>
      <th>...</th>
      <th>OfficeProducts OfficePaperProducts Paper Stationery Notebooks,WritingPads&amp;Diaries CompositionNotebooks</th>
      <th>OfficeProducts OfficePaperProducts Paper Stationery Notebooks,WritingPads&amp;Diaries Notepads&amp;MemoBooks</th>
      <th>OfficeProducts OfficePaperProducts Paper Stationery Notebooks,WritingPads&amp;Diaries WireboundNotebooks</th>
      <th>OfficeProducts OfficePaperProducts Paper Stationery Pens,Pencils&amp;WritingSupplies Pens&amp;Refills BottledInk</th>
      <th>OfficeProducts OfficePaperProducts Paper Stationery Pens,Pencils&amp;WritingSupplies Pens&amp;Refills FountainPens</th>
      <th>OfficeProducts OfficePaperProducts Paper Stationery Pens,Pencils&amp;WritingSupplies Pens&amp;Refills GelInkRollerballPens</th>
      <th>OfficeProducts OfficePaperProducts Paper Stationery Pens,Pencils&amp;WritingSupplies Pens&amp;Refills LiquidInkRollerballPens</th>
      <th>OfficeProducts OfficePaperProducts Paper Stationery Pens,Pencils&amp;WritingSupplies Pens&amp;Refills RetractableBallpointPens</th>
      <th>OfficeProducts OfficePaperProducts Paper Stationery Pens,Pencils&amp;WritingSupplies Pens&amp;Refills StickBallpointPens</th>
      <th>Toys&amp;Games Arts&amp;Crafts Drawing&amp;PaintingSupplies ColouringPens&amp;Markers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B07JW9H4J1</td>
      <td>Wayona Nylon Braided USB to Lightning Fast Cha...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>399.0</td>
      <td>1099.0</td>
      <td>64</td>
      <td>4.2</td>
      <td>24269</td>
      <td>High Compatibility : Compatible With iPhone 12...</td>
      <td>AG3D6O4STAQKAY2UVGEUV46KN35Q,AHMY5CWJMMK5BJRBB...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B098NS6PVG</td>
      <td>Ambrane Unbreakable 60W / 3A Fast Charging 1.5...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>199.0</td>
      <td>349.0</td>
      <td>43</td>
      <td>4.0</td>
      <td>43994</td>
      <td>Compatible with all Type C enabled devices, be...</td>
      <td>AECPFYFQVRUWC3KGNLJIOREFP5LQ,AGYYVPDD7YG7FYNBX...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B096MSW6CT</td>
      <td>Sounce Fast Phone Charging Cable &amp; Data Sync U...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>199.0</td>
      <td>1899.0</td>
      <td>90</td>
      <td>3.9</td>
      <td>7928</td>
      <td>【 Fast Charger&amp; Data Sync】-With built-in safet...</td>
      <td>AGU3BBQ2V2DDAMOAKGFAWDDQ6QHA,AESFLDV2PT363T2AQ...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B08HDJ86NZ</td>
      <td>boAt Deuce USB 300 2 in 1 Type-C &amp; Micro USB S...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>329.0</td>
      <td>699.0</td>
      <td>53</td>
      <td>4.2</td>
      <td>94363</td>
      <td>The boAt Deuce USB 300 2 in 1 cable is compati...</td>
      <td>AEWAZDZZJLQUYVOVGBEUKSLXHQ5A,AG5HTSFRRE6NL3M5S...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B08CF3B7N1</td>
      <td>Portronics Konnect L 1.2M Fast Charging 3A 8 P...</td>
      <td>Computers&amp;Accessories|Accessories&amp;Peripherals|...</td>
      <td>154.0</td>
      <td>399.0</td>
      <td>61</td>
      <td>4.2</td>
      <td>16905</td>
      <td>[CHARGE &amp; SYNC FUNCTION]- This cable comes wit...</td>
      <td>AE3Q6KSUK5P75D5HFYHCRAOLODSA,AFUGIFH5ZAFXRDSZH...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 229 columns</p>
</div>



### 使用PCA降维


```python
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

X = data.drop(columns=['product_id', 'product_name', 'category', 'about_product', 'product_id', 'user_id', 'user_name', 'review_id', 'review_title', 'review_content', 'img_link', 'product_link', 'categories', 'rating', 'rating_count', 0])
Y = data['rating']
X = data[['discounted_price', 'actual_price', 'discount_percentage']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>discounted_price</th>
      <th>actual_price</th>
      <th>discount_percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1451</th>
      <td>1499.00</td>
      <td>2199.0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>979.00</td>
      <td>1999.0</td>
      <td>51</td>
    </tr>
    <tr>
      <th>254</th>
      <td>96.00</td>
      <td>399.0</td>
      <td>76</td>
    </tr>
    <tr>
      <th>1167</th>
      <td>2698.00</td>
      <td>3945.0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1239</th>
      <td>1599.00</td>
      <td>1999.0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1130</th>
      <td>3249.00</td>
      <td>6295.0</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>998.06</td>
      <td>1282.0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>860</th>
      <td>499.00</td>
      <td>1299.0</td>
      <td>62</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>199.00</td>
      <td>999.0</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>292.00</td>
      <td>499.0</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
<p>1172 rows × 3 columns</p>
</div>




```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Initialize the decision tree regressor
model = DecisionTreeRegressor(random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Set up GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X_train, Y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Use the best model found by grid search to make predictions
best_model = grid_search.best_estimator_

# Make predictions on the test set
Y_pred = best_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")

```

    Best Parameters: {'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 5}
    Mean Squared Error: 0.07989262918129585



```python
model
```




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeRegressor(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeRegressor</label><div class="sk-toggleable__content"><pre>DecisionTreeRegressor(random_state=42)</pre></div></div></div></div></div>




```python
from sklearn.tree import export_graphviz
feature_names = ['discounted_price', 'actual_price', 'discount_percentage']
class_names = ['rating']
export_graphviz(
    best_model,
    out_file='tree.dot',
    feature_names=feature_names,
    class_names=class_names,
    rounded=True,
    filled=True
)
```


```python
from IPython.display import Image

# Display the PNG image
Image(filename='tree.png')

```




    
![png](output_42_0.png)
    



# 结论    
样本量与误差的关系:
    样本量较大时（如821个样本），均方误差（squared_error）较高（0.107），表明数据变异性较大或模型预测难度增加。
    样本量较小时（如51、58、59个样本），均方误差显著降低（0.037–0.039），可能因数据更一致或模型在特定子群体中表现更好。
    极端情况：样本量为1时，误差极低（0.01），但缺乏统计意义，可能为过拟合或偶然结果。

折扣率的影响 (仅第一个表格):
    当折扣率（discount_percentage）≤ 5.5%时，样本量较大（821），预测值（value=4.126）低于其他分组（如4.243、4.205），可能暗示低折扣率与较低目标值相关。

预测值的分布趋势:
    预测值（value）在样本量适中的分组（如51、58、59个样本）中较高（4.243–4.232），而在样本量较少（如27个样本）或极多（如821个样本）时较低（3.789–4.126），可能反映数据分布的非线性特征。
    
模型稳定性:
    均方误差波动较大（0.01–0.118），表明模型在不同数据子集上的表现差异显著，需进一步优化或验证分组的合理性。


```python

```
