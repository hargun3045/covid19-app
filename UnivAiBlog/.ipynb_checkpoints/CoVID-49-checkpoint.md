# Title
> summary


# Big Brother - Healthcare edition

### Building a classifier using the [fastai](https://www.fast.ai/) library 

```python
from fastai.tabular import *
```

```python
df.head(3)
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
      <th>ID</th>
      <th>age</th>
      <th>sex</th>
      <th>cough</th>
      <th>fever</th>
      <th>chills</th>
      <th>sore_throat</th>
      <th>headache</th>
      <th>fatigue</th>
      <th>urgency_of_admission</th>
      <th>...</th>
      <th>province</th>
      <th>country</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>geo_resolution</th>
      <th>date_onset_symptoms</th>
      <th>date_admission_hospital</th>
      <th>date_confirmation</th>
      <th>date_death_or_discharge</th>
      <th>source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>30.0</td>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Low</td>
      <td>...</td>
      <td>Anhui</td>
      <td>China</td>
      <td>31.646960</td>
      <td>117.716600</td>
      <td>admin3</td>
      <td>2020-01-18</td>
      <td>2020-01-20</td>
      <td>2020-01-22</td>
      <td>NaN</td>
      <td>http://ah.people.com.cn/GB/n2/2020/0127/c35826...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>47.0</td>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Low</td>
      <td>...</td>
      <td>Anhui</td>
      <td>China</td>
      <td>31.778630</td>
      <td>117.331900</td>
      <td>admin3</td>
      <td>2020-01-10</td>
      <td>2020-01-21</td>
      <td>2020-01-23</td>
      <td>NaN</td>
      <td>http://ah.people.com.cn/GB/n2/2020/0127/c35826...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>49.0</td>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Low</td>
      <td>...</td>
      <td>Anhui</td>
      <td>China</td>
      <td>31.828313</td>
      <td>117.224844</td>
      <td>point</td>
      <td>2020-01-15</td>
      <td>2020-01-20</td>
      <td>2020-01-23</td>
      <td>NaN</td>
      <td>http://ah.people.com.cn/GB/n2/2020/0127/c35826...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 23 columns</p>
</div>



## Independent variable

This is the value we want to predict

```python
y_col = 'urgency_of_admission'
```

## Dependent variable

The values on which we can make a prediciton

```python
cat_names = ['sex', 'cough', 'fever', 'chills', 'sore_throat', 'headache', 'fatigue']
```

```python
cont_names = ['age']
```

```python
data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs = procs)
         .split_by_idx(list(range(660,861)))
         .label_from_df(cols=y_col)
         .add_test(test)
         .databunch()   )
```

```python
data.show_batch(rows=10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>sex</th>
      <th>cough</th>
      <th>fever</th>
      <th>chills</th>
      <th>sore_throat</th>
      <th>headache</th>
      <th>fatigue</th>
      <th>age_na</th>
      <th>age</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>male</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>False</td>
      <td>-1.3107</td>
      <td>High</td>
    </tr>
    <tr>
      <td>male</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>False</td>
      <td>-1.1236</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>False</td>
      <td>1.2461</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>female</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>False</td>
      <td>0.4354</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>False</td>
      <td>-0.0635</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>female</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>False</td>
      <td>1.5579</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>female</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>False</td>
      <td>1.7449</td>
      <td>High</td>
    </tr>
    <tr>
      <td>female</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>False</td>
      <td>0.5601</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>False</td>
      <td>-0.7495</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>female</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>False</td>
      <td>-0.9989</td>
      <td>Low</td>
    </tr>
  </tbody>
</table>


## Model

Here we build our machine learning model that will learn from the dataset to classify between patients

```python
learn = tabular_learner(data, layers = [200,100], metrics = accuracy)
```

```python
testdf.urgency_of_admission.value_counts()
```




    Low     180
    High     21
    Name: urgency_of_admission, dtype: int64



### Making predictions

We've taken out a test set to see how well our model works, by making predictions on them.

Interestingly, all those predicted with 'High' urgency have a common trait of absence of **chills** and **sore throat**

```python
testdf.predictions.value_counts()
```




    Low     183
    High     18
    Name: predictions, dtype: int64



```python
testdf[testdf.predictions == 'High'][['predictions','age','sex','cough','fever','chills','sore_throat','headache']].iloc[:13]
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
      <th>predictions</th>
      <th>age</th>
      <th>sex</th>
      <th>cough</th>
      <th>fever</th>
      <th>chills</th>
      <th>sore_throat</th>
      <th>headache</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>666</td>
      <td>High</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>680</td>
      <td>High</td>
      <td>50.0</td>
      <td>male</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>701</td>
      <td>High</td>
      <td>60.0</td>
      <td>male</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>703</td>
      <td>High</td>
      <td>18.0</td>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>712</td>
      <td>High</td>
      <td>0.0</td>
      <td>male</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>731</td>
      <td>High</td>
      <td>20.0</td>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>733</td>
      <td>High</td>
      <td>NaN</td>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>743</td>
      <td>High</td>
      <td>20.0</td>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>746</td>
      <td>High</td>
      <td>NaN</td>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>747</td>
      <td>High</td>
      <td>NaN</td>
      <td>male</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>760</td>
      <td>High</td>
      <td>12.0</td>
      <td>female</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>762</td>
      <td>High</td>
      <td>18.0</td>
      <td>female</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <td>771</td>
      <td>High</td>
      <td>20.0</td>
      <td>female</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>


