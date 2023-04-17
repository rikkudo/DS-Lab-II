**1. Import CDN Dataset**


```python

```

    /usr/local/lib/python3.9/dist-packages/gdown/cli.py:121: FutureWarning: Option `--id` was deprecated in version 4.3.1 and will be removed in 5.0. You don't need to pass it anymore to use a file ID.
      warnings.warn(
    Downloading...
    From: https://drive.google.com/uc?id=14TMBl6tfV2BZ1QQZCs0HeasngNLmnUUn
    To: /content/data.xlsx
    100% 17.4M/17.4M [00:00<00:00, 69.7MB/s]
    

**2. Import Library**


```python
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import ipywidgets as widgets

# For visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
pd.options.display.max_rows = None
pd.options.display.max_columns = None
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn import preprocessing, svm
import tensorflow as tf 
import sys
import missingno as mno
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
```

**3. Explore Data**


```python
data_xls = pd.read_excel('data.xlsx', 'cdn_customer_qoe_anon', dtype=str, index_col=None)
data_xls.to_csv('data.csv', encoding='utf-8', index=False)
```

In the dataset, there are two column, "Start Time" and "End Time" contain datetime data. Parse this two column into datetime format for Time Series Analysis and check the amount of data on in the sequence.


```python
df = pd.read_csv("data.csv", parse_dates=["Start Time", "End Time"])
df.head()
```





  <div id="df-fcef91fc-9ad4-4652-b288-10c14c4f9719">
    <div class="colab-df-container">
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
      <th>Column1</th>
      <th>Start Time</th>
      <th>Playtime</th>
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>CDN Node Host</th>
      <th>Connection Type</th>
      <th>Device</th>
      <th>Device Type</th>
      <th>Browser</th>
      <th>Browser Version</th>
      <th>OS</th>
      <th>OS Version</th>
      <th>Device ID</th>
      <th>Happiness Value</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>End Time</th>
      <th>Crash Status</th>
      <th>End of Playback Status</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Program_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>Content_TV_Show_N</th>
      <th>Country_N</th>
      <th>City_N</th>
      <th>Region_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2022-07-12 00:00:14</td>
      <td>11</td>
      <td>10</td>
      <td>0</td>
      <td>0.879</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Ethernet-100</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>Android</td>
      <td>Android 10</td>
      <td>186ba46d-4400-39df-8693-1ca9d25caf48</td>
      <td>Smile (7-8.5)</td>
      <td>7.393</td>
      <td>0.0</td>
      <td>0</td>
      <td>19504</td>
      <td>2022-10-04 00:00:26</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>564</td>
      <td>784</td>
      <td>0</td>
      <td>16</td>
      <td>64</td>
      <td>2672</td>
      <td>3</td>
      <td>263</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2022-07-12 00:00:38</td>
      <td>73</td>
      <td>72</td>
      <td>0</td>
      <td>1.170</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>Android</td>
      <td>Android 10</td>
      <td>35f76782-6990-3fa2-bdb8-0f35a2c5569c</td>
      <td>Happy (8.5-10)</td>
      <td>9.399</td>
      <td>0.0</td>
      <td>0</td>
      <td>19033</td>
      <td>2022-10-04 00:01:52</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>480</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2022-07-12 00:02:02</td>
      <td>21</td>
      <td>20</td>
      <td>0</td>
      <td>1.133</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>Android</td>
      <td>Android 10</td>
      <td>16846b9d-4bd6-3b16-a464-cb99378f3249</td>
      <td>Neutral (5-7)</td>
      <td>6.999</td>
      <td>0.0</td>
      <td>0</td>
      <td>19071</td>
      <td>2022-10-04 00:02:24</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>346</td>
      <td>786</td>
      <td>0</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2022-07-12 00:02:24</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>Android</td>
      <td>Android 10</td>
      <td>16846b9d-4bd6-3b16-a464-cb99378f3249</td>
      <td>Angry (0-3)</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2022-10-04 00:02:26</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>346</td>
      <td>997</td>
      <td>0</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2022-07-12 00:02:25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>Android</td>
      <td>Android 10</td>
      <td>16846b9d-4bd6-3b16-a464-cb99378f3249</td>
      <td>Angry (0-3)</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2022-10-04 00:02:28</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>346</td>
      <td>997</td>
      <td>0</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fcef91fc-9ad4-4652-b288-10c14c4f9719')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-fcef91fc-9ad4-4652-b288-10c14c4f9719 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fcef91fc-9ad4-4652-b288-10c14c4f9719');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.tail()
```





  <div id="df-f944dfd8-b95e-43a4-879a-09811692da83">
    <div class="colab-df-container">
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
      <th>Column1</th>
      <th>Start Time</th>
      <th>Playtime</th>
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>CDN Node Host</th>
      <th>Connection Type</th>
      <th>Device</th>
      <th>Device Type</th>
      <th>Browser</th>
      <th>Browser Version</th>
      <th>OS</th>
      <th>OS Version</th>
      <th>Device ID</th>
      <th>Happiness Value</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>End Time</th>
      <th>Crash Status</th>
      <th>End of Playback Status</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Program_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>Content_TV_Show_N</th>
      <th>Country_N</th>
      <th>City_N</th>
      <th>Region_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>102251</th>
      <td>102251</td>
      <td>2022-07-25 23:06:05</td>
      <td>15282</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>Mobile</td>
      <td>iPhone</td>
      <td>SmartPhone</td>
      <td>Mobile Safari</td>
      <td>Mobile Safari</td>
      <td>iOS</td>
      <td>iOS iOS:15.2</td>
      <td>341C9AAD-8C51-4892-B3A8-E83EFE85439E</td>
      <td>Angry (0-3)</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2022-10-18 03:20:47</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>570</td>
      <td>1504</td>
      <td>0</td>
      <td>2</td>
      <td>153</td>
      <td>2434</td>
      <td>3</td>
      <td>367</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102252</th>
      <td>102252</td>
      <td>2022-07-25 22:55:39</td>
      <td>16582</td>
      <td>16581</td>
      <td>0</td>
      <td>0.990</td>
      <td>0.000</td>
      <td>11377663</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>Android</td>
      <td>Android 10</td>
      <td>68c8ec91-d840-3939-be6f-b47410fc40c2</td>
      <td>Happy (8.5-10)</td>
      <td>9.998</td>
      <td>0.0</td>
      <td>0</td>
      <td>18191</td>
      <td>2022-10-18 03:32:02</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>475</td>
      <td>1014</td>
      <td>0</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>39</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102253</th>
      <td>102253</td>
      <td>2022-07-25 23:09:33</td>
      <td>21166</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android</td>
      <td>STBAndroid</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>Android</td>
      <td>Android 10</td>
      <td>5b63edc5-6b14-3d17-b2b3-654b438d14cb</td>
      <td>Angry (0-3)</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2022-10-18 05:02:21</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>249</td>
      <td>1076</td>
      <td>0</td>
      <td>16</td>
      <td>41</td>
      <td>2672</td>
      <td>3</td>
      <td>56</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102254</th>
      <td>102254</td>
      <td>2022-07-25 11:47:37</td>
      <td>65122</td>
      <td>65115</td>
      <td>2</td>
      <td>6.103</td>
      <td>0.001</td>
      <td>NaN</td>
      <td>None</td>
      <td>PC( Windows )</td>
      <td>PC</td>
      <td>Chrome</td>
      <td>Chrome 106.0.0.0</td>
      <td>Windows</td>
      <td>Windows 10</td>
      <td>bc64afea-dafa-4d58-bdd2-60ac8de742c2</td>
      <td>Smile (7-8.5)</td>
      <td>7.465</td>
      <td>0.0</td>
      <td>0</td>
      <td>27550</td>
      <td>2022-10-18 05:53:00</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>622</td>
      <td>1437</td>
      <td>0</td>
      <td>8</td>
      <td>158</td>
      <td>694</td>
      <td>3</td>
      <td>56</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102255</th>
      <td>102255</td>
      <td>2022-07-25 14:07:08</td>
      <td>75837</td>
      <td>75717</td>
      <td>0</td>
      <td>120.000</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android</td>
      <td>STBAndroid</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>Android</td>
      <td>Android 10</td>
      <td>a6d4fc0c-c548-490f-a5aa-d9de96cce74f</td>
      <td>Angry (0-3)</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1</td>
      <td>36285</td>
      <td>2022-10-18 11:11:06</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>101</td>
      <td>902</td>
      <td>0</td>
      <td>13</td>
      <td>15</td>
      <td>2672</td>
      <td>3</td>
      <td>263</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f944dfd8-b95e-43a4-879a-09811692da83')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f944dfd8-b95e-43a4-879a-09811692da83 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f944dfd8-b95e-43a4-879a-09811692da83');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




The dataset consist of 102256 row

Show Minimum and Maximum time in Column Start Time


```python
df["Start Time"].min(), df["Start Time"].max()
```




    (Timestamp('2022-07-12 00:00:14'), Timestamp('2022-07-25 23:59:56'))



Show Minimum and Maximum time in Column End Time


```python
df["End Time"].min(), df["End Time"].max()
```




    (Timestamp('2022-10-04 00:00:26'), Timestamp('2022-10-18 11:11:06'))



Check Shape of data


```python
df.shape
```




    (102256, 33)



From shape of data, it can be seen there are 102256 series on 33 variable

**Show Correlation Matrix**


```python
plt.figure(figsize = (16, 9))
s = sns.heatmap(df.corr(),
                annot = True,
                cmap = 'RdBu',
                vmin = -1,
                vmax = 1)
s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 12)
s.set_xticklabels(s.get_xticklabels(), rotation = 90, fontsize = 12)
plt.title('Correlation Heatmap')
plt.show()
```

    <ipython-input-10-4721ba411dd2>:2: FutureWarning:
    
    The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
    
    


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_19_1.png)
    


From the image it can be seen that Column Playtime and Effective Playtime have high correlation and we can say this 2 columns is similar. 

We choose Playtime because it is ground values for Effective Playtime after have influence from Interruptions, Buffer Ratio and any other correlation variable

**QoE Between Interruptions and Playtime**


```python
plt.plot( df["Interruptions"],df["Playtime"], 'bo')
```




    [<matplotlib.lines.Line2D at 0x7fc4a4fc6250>]




    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_22_1.png)
    


It shows the effect of Interruptions on the Playtime that have direct impact to Effective Playtime and it also shows outliers there

**Show important Value from every column**


```python
# null values
null_values = df.isnull().sum()
df_nulls = pd.DataFrame({'Column':null_values.index, 'Nulls':null_values.values})

# Count non empty cells
df_count = df_nulls
df_count = df_count.rename(columns={"Nulls": "Count"})
nb_rows = df.shape[0] # 102256
df_count.iloc[:,1] = nb_rows - df_count.iloc[:,1]

# count unique cells
uniqueValues = df.nunique()
df_unique = pd.DataFrame({'Column':uniqueValues.index, 'Unique':uniqueValues.values})

# data types
dty = df.dtypes
df_dty = pd.DataFrame({'Column':dty.index, 'Type':dty.values})
```


```python
# merge all together in one dataframe
analyzed_df = pd.concat([df_nulls.iloc[:,0], df_count.iloc[:,1], df_nulls.iloc[:,1], df_unique.iloc[:,1], df_dty.iloc[:,1]], axis = 1)
print(analyzed_df)
```

                        Column   Count   Nulls  Unique            Type
    0                  Column1  102256       0  102256           int64
    1               Start Time  102256       0   96503  datetime64[ns]
    2                 Playtime  102256       0    4752           int64
    3       Effective Playtime  102256       0    4440           int64
    4            Interruptions  102256       0      43           int64
    5                Join Time  102256       0    5554         float64
    6             Buffer Ratio  102256       0    2076         float64
    7            CDN Node Host   36979   65277     342          object
    8          Connection Type  102256       0      19          object
    9                   Device  102256       0      15          object
    10             Device Type  102256       0      13          object
    11                 Browser  102256       0      15          object
    12         Browser Version  101411     845      64          object
    13                      OS  102256       0       9          object
    14              OS Version  102256       0      78          object
    15               Device ID  102256       0    1692          object
    16         Happiness Value  102255       1       5          object
    17         Happiness Score  102256       0    6534         float64
    18         Playback Stalls  102256       0     425         float64
    19   Startup Error (Count)  102256       0       2           int64
    20                 Latency  102256       0   13416           int64
    21                End Time  102256       0   96251  datetime64[ns]
    22            Crash Status    1845  100411       2          object
    23  End of Playback Status  102255       1       4          object
    24               User_ID_N  102256       0     700           int64
    25                 Title_N  102256       0    1639           int64
    26               Program_N  102256       0       1           int64
    27         Device_Vendor_N  102256       0      25           int64
    28          Device_Model_N  102256       0     164           int64
    29       Content_TV_Show_N  102256       0    2747           int64
    30               Country_N  102256       0      15           int64
    31                  City_N  102256       0     406           int64
    32                Region_N  102256       0       2           int64
    

From the result, it can be seen we have several columns containing missing value. It can also be seen column Programn_N just have 1 unique values and we can simply say it adds no value, all Zeros.

We also have both numerical and categorical data, for categorical data, we will encode it using Label Encoding

**3.1 Handling Missing Values**

Find several missing values in columns [CDN Node Host and Browser Version], Drop those columns because they have many missing value. For other column, fill the missing values with other values because we still need them.

Also delete column1 and Program_N.

Delete  Content_TV_Show_N due to the high cardinality.

Delete End Time since we will just use Start Time.

Delete Happiness Value' as Happiness score exists.

Delete Effective Playtime since it came from Interruptions and Playtime


```python
df = df.drop(["Column1", "Effective Playtime", "CDN Node Host", "Browser Version", "Program_N", "End Time", "Happiness Value"], axis = 1)

#Fill Crash Status Column
df["Crash Status"] = df["Crash Status"].astype('category')
df["Crash Status"] = df["Crash Status"].cat.add_categories("No Error Crash").fillna("No Error Crash")
df["Crash Status"] = df["Crash Status"].astype('object')

#fill Happiness Value Column
# df["Happiness Value"] = df["Happiness Value"].astype('category')
# df["Happiness Value"] = df["Happiness Value"].fillna("Angry (0-3)")
# df["Happiness Value"] = df["Happiness Value"].astype('object')

#Fill End Of Playback Status Column
df["End of Playback Status"] = df["End of Playback Status"].astype('category')
df["End of Playback Status"] = df["End of Playback Status"].fillna("On Stop")
df["End of Playback Status"] = df["End of Playback Status"].astype('object')

df.isna().sum()
```




    Start Time                0
    Playtime                  0
    Interruptions             0
    Join Time                 0
    Buffer Ratio              0
    Connection Type           0
    Device                    0
    Device Type               0
    Browser                   0
    OS                        0
    OS Version                0
    Device ID                 0
    Happiness Score           0
    Playback Stalls           0
    Startup Error (Count)     0
    Latency                   0
    Crash Status              0
    End of Playback Status    0
    User_ID_N                 0
    Title_N                   0
    Device_Vendor_N           0
    Device_Model_N            0
    Content_TV_Show_N         0
    Country_N                 0
    City_N                    0
    Region_N                  0
    dtype: int64



There are no more missing values


```python
df.shape
```




    (102256, 26)



It can be seen after removing several columns, now the dataset just have 26 columns or variable there

**3.2 Preprocessing Data**


```python
df.set_index("Start Time", inplace=True)
```

Perform Label Encoding for Categorical Data


```python
for i in df.select_dtypes('object').columns:
  le = LabelEncoder().fit(df[i])
  df[i] = le.transform(df[i]) 

df.head()
```





  <div id="df-5ef8a8eb-88d7-4d59-b623-63341282f335">
    <div class="colab-df-container">
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
      <th>Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>Connection Type</th>
      <th>Device</th>
      <th>Device Type</th>
      <th>Browser</th>
      <th>OS</th>
      <th>OS Version</th>
      <th>Device ID</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>Crash Status</th>
      <th>End of Playback Status</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>Content_TV_Show_N</th>
      <th>Country_N</th>
      <th>City_N</th>
      <th>Region_N</th>
    </tr>
    <tr>
      <th>Start Time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-07-12 00:00:14</th>
      <td>11</td>
      <td>0</td>
      <td>0.879</td>
      <td>0.0</td>
      <td>7</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>209</td>
      <td>7.393</td>
      <td>0.0</td>
      <td>0</td>
      <td>19504</td>
      <td>1</td>
      <td>3</td>
      <td>564</td>
      <td>784</td>
      <td>16</td>
      <td>64</td>
      <td>2672</td>
      <td>3</td>
      <td>263</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-07-12 00:00:38</th>
      <td>73</td>
      <td>0</td>
      <td>1.170</td>
      <td>0.0</td>
      <td>16</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>387</td>
      <td>9.399</td>
      <td>0.0</td>
      <td>0</td>
      <td>19033</td>
      <td>1</td>
      <td>3</td>
      <td>480</td>
      <td>1</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-07-12 00:02:02</th>
      <td>21</td>
      <td>0</td>
      <td>1.133</td>
      <td>0.0</td>
      <td>16</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>197</td>
      <td>6.999</td>
      <td>0.0</td>
      <td>0</td>
      <td>19071</td>
      <td>1</td>
      <td>3</td>
      <td>346</td>
      <td>786</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-07-12 00:02:24</th>
      <td>1</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>16</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>197</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>346</td>
      <td>997</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2022-07-12 00:02:25</th>
      <td>1</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>16</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>197</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>346</td>
      <td>997</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5ef8a8eb-88d7-4d59-b623-63341282f335')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5ef8a8eb-88d7-4d59-b623-63341282f335 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5ef8a8eb-88d7-4d59-b623-63341282f335');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




**3.3 Exploratory Data Analysis (EDA)**

Show the distribution of data in every columns based on histogram and density


```python
fig, axes = plt.subplots(len(df.columns)//3, 3, figsize=(14, 90))

i = 0
for triaxis in axes:
    for axis in triaxis:
        df.hist(column = df.columns[i], bins = 100, ax=axis)
        i = i+1
```


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_40_0.png)
    



```python
df.plot(kind='density',subplots=True,layout=(10,5),figsize=(12,12),sharex=False)
plt.show()
```


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_41_0.png)
    


From the images it can be seen that the dataset have so many outliers on each features. Checking the outliers using distribution of data


```python
df.describe()
```





  <div id="df-11eef46e-87b3-48f1-ab54-4b2d4da6e231">
    <div class="colab-df-container">
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
      <th>Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>Connection Type</th>
      <th>Device</th>
      <th>Device Type</th>
      <th>Browser</th>
      <th>OS</th>
      <th>OS Version</th>
      <th>Device ID</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>Crash Status</th>
      <th>End of Playback Status</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>Content_TV_Show_N</th>
      <th>Country_N</th>
      <th>City_N</th>
      <th>Region_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
      <td>102256.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>328.965479</td>
      <td>0.099544</td>
      <td>1.158462</td>
      <td>0.261755</td>
      <td>12.391116</td>
      <td>2.391889</td>
      <td>8.690757</td>
      <td>0.859470</td>
      <td>0.831179</td>
      <td>5.934243</td>
      <td>724.722393</td>
      <td>5.166843</td>
      <td>0.003805</td>
      <td>0.013564</td>
      <td>13360.818788</td>
      <td>1.006914</td>
      <td>2.956638</td>
      <td>392.976686</td>
      <td>809.658494</td>
      <td>13.127699</td>
      <td>69.461704</td>
      <td>2421.410568</td>
      <td>3.933021</td>
      <td>150.772669</td>
      <td>0.027676</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1799.357645</td>
      <td>12.007473</td>
      <td>2.843884</td>
      <td>3.568773</td>
      <td>4.246433</td>
      <td>3.569991</td>
      <td>2.438083</td>
      <td>2.354904</td>
      <td>1.921155</td>
      <td>14.757550</td>
      <td>513.058420</td>
      <td>4.381398</td>
      <td>0.202198</td>
      <td>0.115673</td>
      <td>23550.856009</td>
      <td>0.134147</td>
      <td>0.297459</td>
      <td>161.309073</td>
      <td>527.743379</td>
      <td>4.624751</td>
      <td>33.347175</td>
      <td>631.704165</td>
      <td>2.546806</td>
      <td>107.740597</td>
      <td>0.164043</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.583000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>197.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>295.000000</td>
      <td>261.000000</td>
      <td>13.000000</td>
      <td>63.000000</td>
      <td>2672.000000</td>
      <td>3.000000</td>
      <td>76.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>0.790000</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>696.000000</td>
      <td>6.646500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>17862.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>383.000000</td>
      <td>997.000000</td>
      <td>15.000000</td>
      <td>64.000000</td>
      <td>2672.000000</td>
      <td>3.000000</td>
      <td>76.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>90.000000</td>
      <td>0.000000</td>
      <td>1.302000</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1195.000000</td>
      <td>9.607000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>19235.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>487.000000</td>
      <td>1170.250000</td>
      <td>16.000000</td>
      <td>64.000000</td>
      <td>2672.000000</td>
      <td>3.000000</td>
      <td>240.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>86404.000000</td>
      <td>3786.000000</td>
      <td>120.000000</td>
      <td>100.000000</td>
      <td>18.000000</td>
      <td>14.000000</td>
      <td>12.000000</td>
      <td>14.000000</td>
      <td>8.000000</td>
      <td>77.000000</td>
      <td>1691.000000</td>
      <td>10.000000</td>
      <td>44.408000</td>
      <td>1.000000</td>
      <td>359477.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>699.000000</td>
      <td>1638.000000</td>
      <td>24.000000</td>
      <td>163.000000</td>
      <td>2746.000000</td>
      <td>14.000000</td>
      <td>405.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-11eef46e-87b3-48f1-ab54-4b2d4da6e231')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-11eef46e-87b3-48f1-ab54-4b2d4da6e231 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-11eef46e-87b3-48f1-ab54-4b2d4da6e231');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




It can be seen there is outlier in the dataset, plot the data to get a better view


```python
df.boxplot(fontsize=20,rot=90,figsize=(20,10),patch_artist=True)
```




    <Axes: >




    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_45_1.png)
    


From the graph, it can be seen that the difference of data distribution between each features are really extreme because of the outliers

**Data resampling**

The time series dataset has many data points that may be difficult to analyze and visualize. We resample the time by compressing and aggregating it to daily intervals, since we just have 2 weeks time window. We will have fewer data points that are easier to view data behaviour.

The resample() method will aggregate all the data points in the time series and change them to daily intervals.

With resampling data we also can view the time series data like performing with moving average because it works with calculating the mean value with Day parameter, because we just have time series data for 14 days


```python
df.resample('D').mean().plot(subplots=True, figsize=(12,12))
```




    array([<Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>, <Axes: xlabel='Start Time'>,
           <Axes: xlabel='Start Time'>], dtype=object)




    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_49_1.png)
    


From the graph, it can be seen resample data compress dataset by average day by day. But this resample data not represent whole of the data since it just take mean values from a day to day. Many important value missing such as outliers. 

To make a better view we plot the graph to compare raw data with resample data


```python
res = df.resample('D').mean()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['Playtime'], label='Actual Data', color='green', linewidth=2)

# Plot the second line graph on the same axes
ax.plot(res['Playtime'], label='Resampling Data', color='blue', linewidth=2)

# Add a legend to the graph
ax.legend()

# Add axis labels and a title
ax.set_xlabel('Date Time')
ax.set_ylabel('Value')
ax.set_title('Playtime')

# Show the graph
plt.show()
```


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_51_0.png)
    


From the graph it can be seen resample data not really representing all of the data, it just show data in the time series using average values within certain period of time. So many values missing there

Because of this problem we will not use resample data in the following steps 

**4. Time-Series Decomposition**

The time series dataset is thought to consist of three systematic components including level, trend, seasonality, and one non-systematic component called noise.

These components are defined as follows:
*   Level: The average value in the series.
*   Trend: The increasing or decreasing value in the series.
*   Seasonality: The repeating short-term cycle in the series.
*   Noise: The random variation in the series.

Decomposition is primarily used for time series analysis, and as an analysis tool it can be used to inform forecasting models on your problem.

It provides a structured way of thinking about a time series forecasting problem, both generally in terms of modeling complexity and specifically in terms of how to best capture each of these components in a given model.

The dataset are messy and noisy. There may be additive and multiplicative components. There may be an increasing trend followed by a decreasing trend. There may be non-repeating cycles mixed in with the repeating seasonality components.

In this section, exploring more about the dataset by decomposition method, we making new variable for a pair of column Playtime and Start Time. Choose Playtime because it represent main feature from time series data. in the following section we will focusing in this for the Time Series analysis.


```python
ts = df.groupby("Start Time")["Playtime"].sum().rename("playing_time")
ts.head()
```




    Start Time
    2022-07-12 00:00:14     11
    2022-07-12 00:00:15    241
    2022-07-12 00:00:25    124
    2022-07-12 00:00:38     73
    2022-07-12 00:01:52    254
    Name: playing_time, dtype: int64




```python
ts.plot()
```




    <Axes: xlabel='Start Time'>




    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_58_1.png)
    


Choose periode 7 for weekly period dataset since the dataset just consist of 2 weeks time series data 


```python
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(ts, period=7)
result.plot()
plt.show()
```


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_60_0.png)
    


From the graph it can be seen that the dataset is not Trend or Seasonal time-series, it is highly likely a residual time series. To know more about it, we will digging it further and break it down one-by-one

**4.1 Trend Analysis**

The trend is the component of a time series that represents variations of low frequency in a time series, the high and medium frequency fluctuations having been filtered out.

The objective of this analysis is to understand if there is a trend in the data and whether this pattern is linear or not. The best tool for this job is visualization.


```python
'''
Plot ts with rolling mean and 95% confidence interval with rolling std.
:parameter    
  :param ts: pandas Series    
  :param window: num - for rolling stats
  :param plot_ma: bool - whether plot moving average
  :param plot_intervals: bool - whether plot upper and lower bounds
'''
def plot_ts(ts, plot_ma=True, plot_intervals=True, window=30,
            figsize=(15,5)):    
   rolling_mean = ts.rolling(window=window).mean()    
   rolling_std = ts.rolling(window=window).std()
   plt.figure(figsize=figsize)    
   plt.title(ts.name)    
   plt.plot(ts[window:], label='Actual values', color="black")    
   if plot_ma:        
      plt.plot(rolling_mean, 'g', label='MA'+str(window),
               color="red")    
   if plot_intervals:
      lower_bound = rolling_mean - (1.96 * rolling_std)
      upper_bound = rolling_mean + (1.96 * rolling_std)
   plt.fill_between(x=ts.index, y1=lower_bound, y2=upper_bound,
                    color='lightskyblue', alpha=0.4)
   plt.legend(loc='best')
   plt.grid(True)
   plt.show()
```

When the dataset has 2 weeks day of observation, start a rolling window with half of it, which is 7 days:


```python
plot_ts(ts, window=7)
```

    <ipython-input-27-53b3c5f9e5ce>:17: UserWarning:
    
    color is redundantly defined by the 'color' keyword argument and the fmt string "g" (-> color=(0.0, 0.5, 0.0, 1)). The keyword argument will take precedence.
    
    


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_66_1.png)
    


Check at the red line in the plot, it can be seen a pattern similar with the time series plot


```python
plot_ts(ts, window=14)
```

    <ipython-input-27-53b3c5f9e5ce>:17: UserWarning:
    
    color is redundantly defined by the 'color' keyword argument and the fmt string "g" (-> color=(0.0, 0.5, 0.0, 1)). The keyword argument will take precedence.
    
    


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_68_1.png)
    


When a rolling window of 2 weeks is used it is obvious getting more similar with the original time series itself. 

This is useful in model design as most of the models require to specify whether the trend component exists and whether it is linear (also said “additive”) or non-linear (also said“multiplicative”).

But is a clear, the dataset not have trend component exists and we can assume it is because the dataset itself is residual time series. 

**4.2 Stationarity Test**

A stationary process is a stochastic process whose unconditional joint probability distribution does not change when shifted in time. Consequently, parameters such as mean and variance also do not change over time, therefore stationary time series are easier to forecast.

There are several ways to establish whether a time series is stationary or not, the most common are good old visualization, looking at the autocorrelation and running statistical tests.

The most common Autocorrelation test is the Dickey-Fuller test (also called “ADF test”) where the null hypothesis is that the time series has a unit root, in other words, that the time series is not stationary.


```python
from statsmodels.tsa.stattools import adfuller

print(" > Is the data stationary ?")
dftest = adfuller(ts, autolag='AIC')
print("Test statistic = {:.3f}".format(dftest[0]))
print("P-value = {:.3f}".format(dftest[1]))
print("Critical values :")
for k, v in dftest[4].items():
    print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))
```

     > Is the data stationary ?
    Test statistic = -56.426
    P-value = 0.000
    Critical values :
    	1%: -3.4304177827253977 - The data is  stationary with 99% confidence
    	5%: -2.861569958890621 - The data is  stationary with 95% confidence
    	10%: -2.5667859460712474 - The data is  stationary with 90% confidence
    

From the result it seems the data is stationary, not residual

**4.3 Autocorrelation plots (ACF & PACF)**

Autocorrelation is Correlation which calculated between the variable and itself at previous time steps. 

An autocorrelation (ACF) plot represents the autocorrelation of the series with lags of itself.

A partial autocorrelation (PACF) plot represents the amount of correlation between a series and a lag of itself that is not explained by correlations at all lower-order lags.


```python
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as smt

def test_stationarity_acf_pacf(ts, sample=0.20, maxlag=30, figsize=(15,10)):
    with plt.style.context(style='bmh'):
        ## set figure
        fig = plt.figure(figsize=figsize)
        ts_ax = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=2)
        pacf_ax = plt.subplot2grid(shape=(2,2), loc=(1,0))
        acf_ax = plt.subplot2grid(shape=(2,2), loc=(1,1))
        
        ## plot ts with mean/std of a sample from the first x% 
        dtf_ts = ts.to_frame(name="ts")
        sample_size = int(len(ts)*sample)
        dtf_ts["mean"] = dtf_ts["ts"].head(sample_size).mean()
        dtf_ts["lower"] = dtf_ts["ts"].head(sample_size).mean() + dtf_ts["ts"].head(sample_size).std()
        dtf_ts["upper"] = dtf_ts["ts"].head(sample_size).mean() - dtf_ts["ts"].head(sample_size).std()
        dtf_ts["ts"].plot(ax=ts_ax, color="black", legend=False)
        dtf_ts["mean"].plot(ax=ts_ax, legend=False, color="red",
                            linestyle="--", linewidth=0.7)
        ts_ax.fill_between(x=dtf_ts.index, y1=dtf_ts['lower'], 
                y2=dtf_ts['upper'], color='lightskyblue', alpha=0.4)
        dtf_ts["mean"].head(sample_size).plot(ax=ts_ax,
                legend=False, color="red", linewidth=0.9)
        ts_ax.fill_between(x=dtf_ts.head(sample_size).index, 
                           y1=dtf_ts['lower'].head(sample_size), 
                           y2=dtf_ts['upper'].head(sample_size),
                           color='lightskyblue')
        
        ## test stationarity (Augmented Dickey-Fuller)
        adfuller_test = sm.tsa.stattools.adfuller(ts, maxlag=maxlag,
                                                  autolag="AIC")
        adf, p, critical_value = adfuller_test[0], adfuller_test[1], adfuller_test[4]["5%"]
        p = round(p, 3)
        conclusion = "Stationary" if p < 0.05 else "Non-Stationary"
        ts_ax.set_title('Dickey-Fuller Test 95%: '+conclusion+
                        '(p value: '+str(p)+')')
        
        ## pacf (for AR) e acf (for MA) 
        smt.plot_pacf(ts, lags=maxlag, ax=pacf_ax, 
                 title="Partial Autocorrelation (for AR component)")
        smt.plot_acf(ts, lags=maxlag, ax=acf_ax,
                 title="Autocorrelation (for MA component)")
        plt.tight_layout()
```


```python
test_stationarity_acf_pacf(ts, sample=0.20, maxlag=30)
```

    /usr/local/lib/python3.9/dist-packages/statsmodels/graphics/tsaplots.py:348: FutureWarning:
    
    The default method 'yw' can produce PACF values outside of the [-1,1] interval. After 0.13, the default will change tounadjusted Yule-Walker ('ywm'). You can use this method now by setting method='ywm'.
    
    


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_78_1.png)
    


It can also be seen, there is no spikes above the blue region, meaning there is no correlation between series and lag itself.

**4.4 Seasonality Analysis**

The seasonal component is that part of the variations in a time series representing 2 weeks of fluctuations with respect to timing, direction and magnitude.

The objective of this last section is to understand what kind of seasonality is affecting the data (weekly seasonality if it presents fluctuations every 7 days.


```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts, period=7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid   
fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False)
ax[0].plot(ts)
ax[0].set_title('Original')
ax[0].grid(True) 
ax[1].plot(trend)
ax[1].set_title('Trend')
ax[1].grid(True)  
ax[2].plot(seasonal)
ax[2].set_title('Seasonality')
ax[2].grid(True)  
ax[3].plot(residual)
ax[3].set_title('Residuals')
ax[3].grid(True)

```


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_82_0.png)
    



```python
tss = sm.add_constant(range(len(ts)))
model = sm.OLS(ts, tss).fit()

# Calculate residuals
residuals = model.resid

# Visualize residuals using a scatter plot
plt.scatter(ts.index, residuals)
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residual plot')
plt.show()
```


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_83_0.png)
    


If a time series shows a pattern but does not exhibit seasonality, it could be due to other factors such as trends, cycles, or irregular fluctuations. From graph above, it can be seen that the dataset is not trend or seasonality. So, it can be irregular fluctuation data

Irregular fluctuations, also known as noise or residuals, are random fluctuations in the data that cannot be explained by trends, cycles, or seasonality. Irregular fluctuations can be caused by measurement error, random events, or other factors.

But from stationary test we know that this data is in stationary form, which means that the stationary happend because the dataset have irregural fluctuation over the time and become residual data

**5. Time Series Forecasting**

**5.1 Vector Auto Regression (VAR)**

In this section, we will introduce to one of the most commonly used methods for multivariate time series forecasting – Vector Auto Regression (VAR).

In a VAR algorithm, each variable is a linear function of the past values of itself and the past values of all the other variables. VAR is able to understand and use the relationship between several variables. This is useful for describing the dynamic behavior of the data and also provides better forecasting results. Additionally, implementing VAR is as simple as using any other univariate technique.

With VAR models, it is possible to elucidate the values of endogenous variables by considering their previously observed values


```python
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

data = df

# Check for stationarity of the time series
def adf_test(series, signif=0.05):
    result = adfuller(series, autolag='AIC')
    pvalue = result[1]
    if pvalue <= signif:
        return True
    else:
        return False

# If any of the series are not stationary, apply differencing until they become stationary
non_stationary_variables = [var for var in data.columns if not adf_test(data[var])]

while len(non_stationary_variables) > 0:
    data = data.diff().dropna()
    non_stationary_variables = [var for var in data.columns if not adf_test(data[var])]

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Fit the VAR model
model = VAR(train_data)
best_aic = np.inf
best_order = None

# Find the best order (p) for the VAR model
for p in range(1, 10):
    result = model.fit(p)
    if result.aic < best_aic:
        best_aic = result.aic
        best_order = p

# Fit the VAR model with the best order
var_model = model.fit(best_order)
print(var_model.summary())

# Forecast the next n steps
n_steps = len(test_data)
forecast = var_model.forecast(train_data.values[-best_order:], steps=n_steps)

# Calculate RMSE and other evaluation metrics if necessary
forecast_df = pd.DataFrame(forecast, index=test_data.index, columns=test_data.columns)
rmse_values = {var: rmse(test_data[var], forecast_df[var]) for var in test_data.columns}
print("RMSE values for each variable: ", rmse_values)

```

    /usr/local/lib/python3.9/dist-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning:
    
    A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
    
    /usr/local/lib/python3.9/dist-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning:
    
    A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.
    
    

    [1;30;43mStreaming output truncated to the last 5000 lines.[0m
    L6.Device                         0.003227         0.010569            0.305           0.760
    L6.Device Type                   -0.008319         0.008641           -0.963           0.336
    L6.Browser                        0.005387         0.012653            0.426           0.670
    L6.OS                             0.005355         0.039462            0.136           0.892
    L6.OS Version                    -0.003898         0.004125           -0.945           0.345
    L6.Device ID                     -0.000022         0.000029           -0.750           0.453
    L6.Happiness Score                0.000915         0.003513            0.261           0.794
    L6.Playback Stalls               -0.082624         0.061181           -1.350           0.177
    L6.Startup Error (Count)         -0.501526         0.295501           -1.697           0.090
    L6.Latency                       -0.000000         0.000001           -0.082           0.935
    L6.Crash Status                   0.365327         0.181246            2.016           0.044
    L6.End of Playback Status        -0.073271         0.078220           -0.937           0.349
    L6.User_ID_N                      0.000292         0.000093            3.145           0.002
    L6.Title_N                       -0.000005         0.000025           -0.210           0.834
    L6.Device_Vendor_N               -0.012350         0.005223           -2.365           0.018
    L6.Device_Model_N                -0.001585         0.000613           -2.584           0.010
    L6.Content_TV_Show_N             -0.000000         0.000025           -0.016           0.987
    L6.Country_N                      0.024503         0.007948            3.083           0.002
    L6.City_N                        -0.000012         0.000149           -0.079           0.937
    L6.Region_N                      -0.236206         0.148237           -1.593           0.111
    L7.Playtime                      -0.000003         0.000007           -0.466           0.641
    L7.Interruptions                 -0.000070         0.000931           -0.075           0.940
    L7.Join Time                     -0.001513         0.004232           -0.357           0.721
    L7.Buffer Ratio                   0.019920         0.004224            4.716           0.000
    L7.Connection Type                0.002871         0.003714            0.773           0.440
    L7.Device                        -0.004764         0.010540           -0.452           0.651
    L7.Device Type                   -0.000882         0.008608           -0.102           0.918
    L7.Browser                       -0.017521         0.012613           -1.389           0.165
    L7.OS                             0.065050         0.039352            1.653           0.098
    L7.OS Version                    -0.004551         0.004115           -1.106           0.269
    L7.Device ID                      0.000013         0.000029            0.441           0.659
    L7.Happiness Score               -0.003283         0.003508           -0.936           0.349
    L7.Playback Stalls               -0.084281         0.061499           -1.370           0.171
    L7.Startup Error (Count)         -0.217775         0.295476           -0.737           0.461
    L7.Latency                        0.000000         0.000001            0.601           0.548
    L7.Crash Status                   0.024423         0.181243            0.135           0.893
    L7.End of Playback Status        -0.023710         0.078209           -0.303           0.762
    L7.User_ID_N                     -0.000052         0.000093           -0.556           0.578
    L7.Title_N                       -0.000034         0.000025           -1.386           0.166
    L7.Device_Vendor_N               -0.002111         0.005206           -0.406           0.685
    L7.Device_Model_N                -0.000277         0.000612           -0.453           0.651
    L7.Content_TV_Show_N             -0.000028         0.000025           -1.148           0.251
    L7.Country_N                      0.001163         0.007926            0.147           0.883
    L7.City_N                         0.000021         0.000149            0.143           0.886
    L7.Region_N                       0.046661         0.147830            0.316           0.752
    L8.Playtime                      -0.000007         0.000007           -1.011           0.312
    L8.Interruptions                  0.000068         0.000931            0.073           0.942
    L8.Join Time                     -0.000778         0.004232           -0.184           0.854
    L8.Buffer Ratio                   0.010573         0.004223            2.504           0.012
    L8.Connection Type               -0.005050         0.003685           -1.370           0.171
    L8.Device                        -0.031781         0.010486           -3.031           0.002
    L8.Device Type                    0.004948         0.008543            0.579           0.562
    L8.Browser                       -0.010480         0.012546           -0.835           0.404
    L8.OS                             0.042387         0.039207            1.081           0.280
    L8.OS Version                     0.001615         0.004103            0.394           0.694
    L8.Device ID                     -0.000008         0.000029           -0.262           0.793
    L8.Happiness Score                0.001046         0.003498            0.299           0.765
    L8.Playback Stalls               -0.084649         0.061501           -1.376           0.169
    L8.Startup Error (Count)          0.265391         0.295266            0.899           0.369
    L8.Latency                       -0.000000         0.000001           -0.316           0.752
    L8.Crash Status                  -0.266788         0.181277           -1.472           0.141
    L8.End of Playback Status         0.076744         0.078190            0.982           0.326
    L8.User_ID_N                     -0.000049         0.000092           -0.527           0.598
    L8.Title_N                       -0.000019         0.000025           -0.790           0.429
    L8.Device_Vendor_N               -0.004744         0.005170           -0.918           0.359
    L8.Device_Model_N                -0.000056         0.000609           -0.092           0.927
    L8.Content_TV_Show_N             -0.000023         0.000024           -0.960           0.337
    L8.Country_N                      0.031474         0.007881            3.994           0.000
    L8.City_N                         0.000115         0.000147            0.782           0.434
    L8.Region_N                      -0.358404         0.147220           -2.434           0.015
    L9.Playtime                      -0.000000         0.000007           -0.028           0.977
    L9.Interruptions                 -0.000156         0.000931           -0.167           0.867
    L9.Join Time                      0.006614         0.004230            1.564           0.118
    L9.Buffer Ratio                   0.018511         0.004222            4.384           0.000
    L9.Connection Type               -0.004868         0.003639           -1.338           0.181
    L9.Device                        -0.002538         0.010382           -0.245           0.807
    L9.Device Type                   -0.002938         0.008423           -0.349           0.727
    L9.Browser                       -0.002299         0.012365           -0.186           0.852
    L9.OS                             0.012145         0.038749            0.313           0.754
    L9.OS Version                    -0.003552         0.004074           -0.872           0.383
    L9.Device ID                     -0.000013         0.000029           -0.467           0.640
    L9.Happiness Score               -0.000905         0.003451           -0.262           0.793
    L9.Playback Stalls               -0.236777         0.061288           -3.863           0.000
    L9.Startup Error (Count)         -0.106004         0.295000           -0.359           0.719
    L9.Latency                       -0.000001         0.000001           -0.947           0.343
    L9.Crash Status                  -0.344620         0.181126           -1.903           0.057
    L9.End of Playback Status        -0.101982         0.078162           -1.305           0.192
    L9.User_ID_N                      0.000056         0.000091            0.618           0.536
    L9.Title_N                        0.000022         0.000025            0.879           0.379
    L9.Device_Vendor_N               -0.000519         0.005103           -0.102           0.919
    L9.Device_Model_N                 0.001269         0.000602            2.109           0.035
    L9.Content_TV_Show_N              0.000002         0.000024            0.097           0.923
    L9.Country_N                      0.009071         0.007826            1.159           0.246
    L9.City_N                        -0.000187         0.000145           -1.285           0.199
    L9.Region_N                       0.059711         0.146277            0.408           0.683
    ============================================================================================
    
    Results for equation Connection Type
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             4.561140         0.654116            6.973           0.000
    L1.Playtime                       0.000003         0.000008            0.387           0.699
    L1.Interruptions                 -0.001403         0.001023           -1.370           0.171
    L1.Join Time                      0.000465         0.004673            0.100           0.921
    L1.Buffer Ratio                  -0.007623         0.004636           -1.644           0.100
    L1.Connection Type                0.131700         0.003999           32.931           0.000
    L1.Device                         0.036067         0.011420            3.158           0.002
    L1.Device Type                   -0.041597         0.009264           -4.490           0.000
    L1.Browser                        0.001621         0.013590            0.119           0.905
    L1.OS                            -0.049640         0.042641           -1.164           0.244
    L1.OS Version                     0.006116         0.004480            1.365           0.172
    L1.Device ID                      0.000329         0.000031           10.508           0.000
    L1.Happiness Score                0.018439         0.003798            4.855           0.000
    L1.Playback Stalls                0.074157         0.067416            1.100           0.271
    L1.Startup Error (Count)         -0.054130         0.324729           -0.167           0.868
    L1.Latency                        0.000001         0.000001            0.915           0.360
    L1.Crash Status                   0.176041         0.198787            0.886           0.376
    L1.End of Playback Status        -0.062560         0.085641           -0.730           0.465
    L1.User_ID_N                      0.000253         0.000100            2.539           0.011
    L1.Title_N                       -0.000173         0.000027           -6.403           0.000
    L1.Device_Vendor_N                0.019160         0.005609            3.416           0.001
    L1.Device_Model_N                -0.001890         0.000661           -2.858           0.004
    L1.Content_TV_Show_N              0.000021         0.000027            0.776           0.438
    L1.Country_N                     -0.013354         0.008604           -1.552           0.121
    L1.City_N                        -0.000093         0.000160           -0.579           0.562
    L1.Region_N                       0.171512         0.160751            1.067           0.286
    L2.Playtime                      -0.000004         0.000008           -0.459           0.646
    L2.Interruptions                 -0.001197         0.001023           -1.170           0.242
    L2.Join Time                      0.005026         0.004674            1.075           0.282
    L2.Buffer Ratio                  -0.009151         0.004647           -1.969           0.049
    L2.Connection Type                0.125110         0.004050           30.892           0.000
    L2.Device                         0.024318         0.011531            2.109           0.035
    L2.Device Type                   -0.036485         0.009390           -3.886           0.000
    L2.Browser                       -0.002137         0.013785           -0.155           0.877
    L2.OS                            -0.045669         0.043107           -1.059           0.289
    L2.OS Version                     0.006964         0.004511            1.544           0.123
    L2.Device ID                      0.000202         0.000032            6.375           0.000
    L2.Happiness Score                0.018601         0.003848            4.834           0.000
    L2.Playback Stalls                0.023392         0.067421            0.347           0.729
    L2.Startup Error (Count)         -0.414261         0.324915           -1.275           0.202
    L2.Latency                       -0.000000         0.000001           -0.463           0.643
    L2.Crash Status                   0.204577         0.199281            1.027           0.305
    L2.End of Playback Status        -0.188893         0.085829           -2.201           0.028
    L2.User_ID_N                      0.000100         0.000101            0.985           0.325
    L2.Title_N                       -0.000148         0.000027           -5.479           0.000
    L2.Device_Vendor_N                0.019266         0.005682            3.390           0.001
    L2.Device_Model_N                 0.000047         0.000669            0.071           0.944
    L2.Content_TV_Show_N              0.000032         0.000027            1.184           0.236
    L2.Country_N                      0.006215         0.008666            0.717           0.473
    L2.City_N                        -0.000065         0.000162           -0.399           0.690
    L2.Region_N                      -0.260068         0.161848           -1.607           0.108
    L3.Playtime                      -0.000013         0.000008           -1.609           0.108
    L3.Interruptions                  0.001355         0.001023            1.324           0.185
    L3.Join Time                     -0.005464         0.004674           -1.169           0.242
    L3.Buffer Ratio                  -0.005835         0.004649           -1.255           0.209
    L3.Connection Type                0.091297         0.004083           22.361           0.000
    L3.Device                         0.019938         0.011590            1.720           0.085
    L3.Device Type                   -0.009338         0.009462           -0.987           0.324
    L3.Browser                        0.007011         0.013862            0.506           0.613
    L3.OS                            -0.066595         0.043261           -1.539           0.124
    L3.OS Version                     0.007624         0.004522            1.686           0.092
    L3.Device ID                      0.000070         0.000032            2.205           0.027
    L3.Happiness Score                0.011885         0.003860            3.079           0.002
    L3.Playback Stalls                0.039438         0.067424            0.585           0.559
    L3.Startup Error (Count)          0.394336         0.324921            1.214           0.225
    L3.Latency                       -0.000000         0.000001           -0.001           0.999
    L3.Crash Status                   0.004843         0.199223            0.024           0.981
    L3.End of Playback Status         0.010318         0.085918            0.120           0.904
    L3.User_ID_N                      0.000008         0.000102            0.081           0.936
    L3.Title_N                       -0.000103         0.000027           -3.818           0.000
    L3.Device_Vendor_N                0.008700         0.005722            1.520           0.128
    L3.Device_Model_N                -0.000322         0.000673           -0.479           0.632
    L3.Content_TV_Show_N             -0.000014         0.000027           -0.525           0.600
    L3.Country_N                      0.012059         0.008716            1.384           0.166
    L3.City_N                        -0.000178         0.000163           -1.088           0.277
    L3.Region_N                      -0.165332         0.162510           -1.017           0.309
    L4.Playtime                      -0.000029         0.000008           -3.594           0.000
    L4.Interruptions                 -0.001014         0.001023           -0.991           0.322
    L4.Join Time                     -0.001849         0.004649           -0.398           0.691
    L4.Buffer Ratio                   0.005019         0.004649            1.080           0.280
    L4.Connection Type                0.071788         0.004099           17.513           0.000
    L4.Device                         0.010860         0.011617            0.935           0.350
    L4.Device Type                   -0.009678         0.009495           -1.019           0.308
    L4.Browser                        0.024015         0.013910            1.727           0.084
    L4.OS                            -0.012145         0.043378           -0.280           0.779
    L4.OS Version                    -0.004189         0.004534           -0.924           0.356
    L4.Device ID                      0.000004         0.000032            0.123           0.902
    L4.Happiness Score                0.009123         0.003864            2.361           0.018
    L4.Playback Stalls               -0.047125         0.067397           -0.699           0.484
    L4.Startup Error (Count)         -0.301753         0.324905           -0.929           0.353
    L4.Latency                        0.000001         0.000001            1.276           0.202
    L4.Crash Status                   0.296021         0.199222            1.486           0.137
    L4.End of Playback Status        -0.026082         0.085937           -0.303           0.762
    L4.User_ID_N                     -0.000107         0.000102           -1.046           0.295
    L4.Title_N                       -0.000019         0.000027           -0.686           0.493
    L4.Device_Vendor_N               -0.000676         0.005742           -0.118           0.906
    L4.Device_Model_N                 0.000235         0.000674            0.349           0.727
    L4.Content_TV_Show_N              0.000016         0.000027            0.601           0.548
    L4.Country_N                      0.007689         0.008738            0.880           0.379
    L4.City_N                         0.000107         0.000164            0.654           0.513
    L4.Region_N                      -0.301448         0.162970           -1.850           0.064
    L5.Playtime                       0.000006         0.000008            0.760           0.447
    L5.Interruptions                 -0.001068         0.001023           -1.044           0.297
    L5.Join Time                      0.000187         0.004651            0.040           0.968
    L5.Buffer Ratio                  -0.005578         0.004642           -1.202           0.229
    L5.Connection Type                0.059661         0.004102           14.543           0.000
    L5.Device                         0.008119         0.011625            0.698           0.485
    L5.Device Type                    0.017414         0.009507            1.832           0.067
    L5.Browser                       -0.014818         0.013931           -1.064           0.287
    L5.OS                             0.021909         0.043422            0.505           0.614
    L5.OS Version                     0.001553         0.004539            0.342           0.732
    L5.Device ID                     -0.000025         0.000032           -0.790           0.430
    L5.Happiness Score                0.001731         0.003864            0.448           0.654
    L5.Playback Stalls                0.006486         0.065628            0.099           0.921
    L5.Startup Error (Count)         -0.179000         0.324874           -0.551           0.582
    L5.Latency                       -0.000000         0.000001           -0.817           0.414
    L5.Crash Status                   0.087377         0.199249            0.439           0.661
    L5.End of Playback Status         0.011911         0.085970            0.139           0.890
    L5.User_ID_N                     -0.000084         0.000102           -0.823           0.410
    L5.Title_N                       -0.000100         0.000027           -3.688           0.000
    L5.Device_Vendor_N                0.002282         0.005746            0.397           0.691
    L5.Device_Model_N                -0.000834         0.000675           -1.235           0.217
    L5.Content_TV_Show_N             -0.000024         0.000027           -0.897           0.370
    L5.Country_N                      0.010884         0.008754            1.243           0.214
    L5.City_N                        -0.000013         0.000164           -0.077           0.939
    L5.Region_N                      -0.224929         0.163191           -1.378           0.168
    L6.Playtime                      -0.000013         0.000008           -1.658           0.097
    L6.Interruptions                 -0.001012         0.001023           -0.989           0.323
    L6.Join Time                      0.000199         0.004653            0.043           0.966
    L6.Buffer Ratio                  -0.001829         0.004642           -0.394           0.694
    L6.Connection Type                0.040448         0.004099            9.868           0.000
    L6.Device                        -0.036303         0.011618           -3.125           0.002
    L6.Device Type                    0.010358         0.009498            1.091           0.275
    L6.Browser                       -0.003130         0.013909           -0.225           0.822
    L6.OS                             0.051437         0.043379            1.186           0.236
    L6.OS Version                    -0.003110         0.004534           -0.686           0.493
    L6.Device ID                     -0.000022         0.000032           -0.697           0.486
    L6.Happiness Score                0.002608         0.003861            0.676           0.499
    L6.Playback Stalls                0.009922         0.067253            0.148           0.883
    L6.Startup Error (Count)          0.782308         0.324829            2.408           0.016
    L6.Latency                        0.000001         0.000001            0.915           0.360
    L6.Crash Status                  -0.513332         0.199234           -2.577           0.010
    L6.End of Playback Status         0.187313         0.085983            2.178           0.029
    L6.User_ID_N                      0.000049         0.000102            0.480           0.631
    L6.Title_N                        0.000003         0.000027            0.110           0.912
    L6.Device_Vendor_N               -0.004393         0.005741           -0.765           0.444
    L6.Device_Model_N                 0.001497         0.000674            2.220           0.026
    L6.Content_TV_Show_N             -0.000035         0.000027           -1.298           0.194
    L6.Country_N                     -0.004218         0.008736           -0.483           0.629
    L6.City_N                         0.000023         0.000164            0.141           0.888
    L6.Region_N                       0.133481         0.162948            0.819           0.413
    L7.Playtime                      -0.000004         0.000008           -0.457           0.647
    L7.Interruptions                 -0.000732         0.001023           -0.716           0.474
    L7.Join Time                      0.001946         0.004652            0.418           0.676
    L7.Buffer Ratio                  -0.002159         0.004643           -0.465           0.642
    L7.Connection Type                0.041973         0.004083           10.281           0.000
    L7.Device                        -0.015272         0.011586           -1.318           0.187
    L7.Device Type                    0.015248         0.009462            1.612           0.107
    L7.Browser                       -0.021325         0.013864           -1.538           0.124
    L7.OS                             0.038776         0.043257            0.896           0.370
    L7.OS Version                    -0.001057         0.004523           -0.234           0.815
    L7.Device ID                     -0.000029         0.000032           -0.908           0.364
    L7.Happiness Score                0.001079         0.003856            0.280           0.780
    L7.Playback Stalls               -0.000884         0.067603           -0.013           0.990
    L7.Startup Error (Count)         -0.182608         0.324801           -0.562           0.574
    L7.Latency                       -0.000000         0.000001           -0.521           0.602
    L7.Crash Status                  -0.070989         0.199231           -0.356           0.722
    L7.End of Playback Status        -0.216036         0.085971           -2.513           0.012
    L7.User_ID_N                     -0.000096         0.000102           -0.937           0.349
    L7.Title_N                       -0.000040         0.000027           -1.492           0.136
    L7.Device_Vendor_N               -0.008495         0.005723           -1.484           0.138
    L7.Device_Model_N                -0.000093         0.000673           -0.139           0.890
    L7.Content_TV_Show_N             -0.000026         0.000027           -0.951           0.342
    L7.Country_N                      0.011891         0.008713            1.365           0.172
    L7.City_N                        -0.000053         0.000163           -0.322           0.747
    L7.Region_N                      -0.134404         0.162501           -0.827           0.408
    L8.Playtime                      -0.000000         0.000008           -0.013           0.990
    L8.Interruptions                  0.001094         0.001023            1.069           0.285
    L8.Join Time                     -0.005789         0.004652           -1.244           0.213
    L8.Buffer Ratio                  -0.002154         0.004642           -0.464           0.643
    L8.Connection Type                0.036428         0.004050            8.994           0.000
    L8.Device                         0.010916         0.011527            0.947           0.344
    L8.Device Type                   -0.001629         0.009391           -0.173           0.862
    L8.Browser                        0.016292         0.013791            1.181           0.237
    L8.OS                            -0.008093         0.043098           -0.188           0.851
    L8.OS Version                    -0.001293         0.004510           -0.287           0.774
    L8.Device ID                     -0.000031         0.000032           -0.987           0.323
    L8.Happiness Score                0.000906         0.003846            0.235           0.814
    L8.Playback Stalls                0.015628         0.067604            0.231           0.817
    L8.Startup Error (Count)          0.088093         0.324570            0.271           0.786
    L8.Latency                       -0.000001         0.000001           -1.398           0.162
    L8.Crash Status                  -0.124665         0.199268           -0.626           0.532
    L8.End of Playback Status         0.060590         0.085950            0.705           0.481
    L8.User_ID_N                      0.000021         0.000101            0.205           0.838
    L8.Title_N                       -0.000055         0.000027           -2.037           0.042
    L8.Device_Vendor_N               -0.003589         0.005683           -0.632           0.528
    L8.Device_Model_N                -0.000032         0.000669           -0.047           0.962
    L8.Content_TV_Show_N              0.000057         0.000027            2.122           0.034
    L8.Country_N                     -0.010883         0.008663           -1.256           0.209
    L8.City_N                         0.000086         0.000162            0.528           0.597
    L8.Region_N                      -0.163077         0.161831           -1.008           0.314
    L9.Playtime                       0.000000         0.000008            0.044           0.965
    L9.Interruptions                  0.001324         0.001023            1.294           0.196
    L9.Join Time                      0.005798         0.004649            1.247           0.212
    L9.Buffer Ratio                  -0.000760         0.004641           -0.164           0.870
    L9.Connection Type                0.039980         0.004000            9.996           0.000
    L9.Device                        -0.008832         0.011412           -0.774           0.439
    L9.Device Type                    0.010513         0.009259            1.135           0.256
    L9.Browser                       -0.020245         0.013592           -1.489           0.136
    L9.OS                             0.011134         0.042595            0.261           0.794
    L9.OS Version                     0.002647         0.004478            0.591           0.554
    L9.Device ID                     -0.000053         0.000031           -1.697           0.090
    L9.Happiness Score               -0.003233         0.003793           -0.852           0.394
    L9.Playback Stalls                0.011273         0.067370            0.167           0.867
    L9.Startup Error (Count)          0.440643         0.324278            1.359           0.174
    L9.Latency                       -0.000000         0.000001           -0.159           0.874
    L9.Crash Status                  -0.322968         0.199102           -1.622           0.105
    L9.End of Playback Status         0.078073         0.085919            0.909           0.364
    L9.User_ID_N                      0.000058         0.000100            0.578           0.563
    L9.Title_N                       -0.000075         0.000027           -2.768           0.006
    L9.Device_Vendor_N                0.009557         0.005609            1.704           0.088
    L9.Device_Model_N                 0.002740         0.000661            4.143           0.000
    L9.Content_TV_Show_N              0.000042         0.000027            1.555           0.120
    L9.Country_N                      0.001978         0.008603            0.230           0.818
    L9.City_N                        -0.000100         0.000160           -0.628           0.530
    L9.Region_N                       0.013917         0.160795            0.087           0.931
    ============================================================================================
    
    Results for equation Device
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             1.095584         0.549745            1.993           0.046
    L1.Playtime                       0.000004         0.000007            0.529           0.597
    L1.Interruptions                 -0.000862         0.000860           -1.002           0.316
    L1.Join Time                     -0.000946         0.003927           -0.241           0.810
    L1.Buffer Ratio                   0.001149         0.003896            0.295           0.768
    L1.Connection Type                0.003926         0.003361            1.168           0.243
    L1.Device                         0.173309         0.009598           18.058           0.000
    L1.Device Type                   -0.004517         0.007786           -0.580           0.562
    L1.Browser                        0.018127         0.011422            1.587           0.112
    L1.OS                            -0.133177         0.035837           -3.716           0.000
    L1.OS Version                     0.009066         0.003765            2.408           0.016
    L1.Device ID                     -0.000012         0.000026           -0.474           0.636
    L1.Happiness Score                0.001343         0.003192            0.421           0.674
    L1.Playback Stalls               -0.059201         0.056659           -1.045           0.296
    L1.Startup Error (Count)          0.748647         0.272914            2.743           0.006
    L1.Latency                        0.000000         0.000000            0.795           0.426
    L1.Crash Status                  -0.631997         0.167068           -3.783           0.000
    L1.End of Playback Status         0.148376         0.071976            2.061           0.039
    L1.User_ID_N                      0.000218         0.000084            2.603           0.009
    L1.Title_N                        0.000002         0.000023            0.087           0.931
    L1.Device_Vendor_N               -0.007566         0.004714           -1.605           0.109
    L1.Device_Model_N                 0.000741         0.000556            1.332           0.183
    L1.Content_TV_Show_N             -0.000158         0.000023           -7.003           0.000
    L1.Country_N                     -0.028006         0.007231           -3.873           0.000
    L1.City_N                        -0.000656         0.000134           -4.887           0.000
    L1.Region_N                       0.105411         0.135101            0.780           0.435
    L2.Playtime                      -0.000004         0.000007           -0.663           0.507
    L2.Interruptions                 -0.000277         0.000860           -0.322           0.748
    L2.Join Time                     -0.003177         0.003929           -0.809           0.419
    L2.Buffer Ratio                   0.003691         0.003906            0.945           0.345
    L2.Connection Type               -0.001228         0.003404           -0.361           0.718
    L2.Device                         0.127680         0.009691           13.175           0.000
    L2.Device Type                   -0.011032         0.007892           -1.398           0.162
    L2.Browser                        0.008882         0.011585            0.767           0.443
    L2.OS                            -0.037560         0.036228           -1.037           0.300
    L2.OS Version                     0.000545         0.003791            0.144           0.886
    L2.Device ID                     -0.000013         0.000027           -0.472           0.637
    L2.Happiness Score                0.003657         0.003234            1.131           0.258
    L2.Playback Stalls                0.067042         0.056663            1.183           0.237
    L2.Startup Error (Count)          0.752077         0.273071            2.754           0.006
    L2.Latency                        0.000001         0.000000            1.445           0.148
    L2.Crash Status                  -0.404565         0.167483           -2.416           0.016
    L2.End of Playback Status         0.227718         0.072134            3.157           0.002
    L2.User_ID_N                      0.000127         0.000085            1.493           0.135
    L2.Title_N                        0.000013         0.000023            0.576           0.564
    L2.Device_Vendor_N               -0.006807         0.004776           -1.425           0.154
    L2.Device_Model_N                 0.000621         0.000562            1.104           0.269
    L2.Content_TV_Show_N             -0.000023         0.000023           -0.998           0.318
    L2.Country_N                      0.005783         0.007283            0.794           0.427
    L2.City_N                        -0.000207         0.000136           -1.521           0.128
    L2.Region_N                      -0.482438         0.136023           -3.547           0.000
    L3.Playtime                      -0.000009         0.000007           -1.304           0.192
    L3.Interruptions                 -0.000132         0.000860           -0.153           0.878
    L3.Join Time                      0.003137         0.003928            0.798           0.425
    L3.Buffer Ratio                  -0.007066         0.003907           -1.808           0.071
    L3.Connection Type                0.003140         0.003431            0.915           0.360
    L3.Device                         0.104449         0.009740           10.723           0.000
    L3.Device Type                   -0.014428         0.007952           -1.814           0.070
    L3.Browser                       -0.008554         0.011650           -0.734           0.463
    L3.OS                            -0.042608         0.036358           -1.172           0.241
    L3.OS Version                     0.004549         0.003801            1.197           0.231
    L3.Device ID                      0.000005         0.000027            0.169           0.865
    L3.Happiness Score                0.005955         0.003244            1.836           0.066
    L3.Playback Stalls                0.064638         0.056666            1.141           0.254
    L3.Startup Error (Count)          0.029760         0.273076            0.109           0.913
    L3.Latency                       -0.000000         0.000000           -0.496           0.620
    L3.Crash Status                   0.037896         0.167434            0.226           0.821
    L3.End of Playback Status         0.087552         0.072209            1.212           0.225
    L3.User_ID_N                      0.000027         0.000086            0.310           0.756
    L3.Title_N                        0.000025         0.000023            1.107           0.268
    L3.Device_Vendor_N               -0.000669         0.004809           -0.139           0.889
    L3.Device_Model_N                -0.001168         0.000565           -2.065           0.039
    L3.Content_TV_Show_N              0.000033         0.000023            1.467           0.142
    L3.Country_N                      0.027674         0.007325            3.778           0.000
    L3.City_N                         0.000045         0.000137            0.329           0.742
    L3.Region_N                      -0.738092         0.136580           -5.404           0.000
    L4.Playtime                       0.000009         0.000007            1.396           0.163
    L4.Interruptions                 -0.000737         0.000860           -0.857           0.391
    L4.Join Time                     -0.001928         0.003907           -0.493           0.622
    L4.Buffer Ratio                   0.007225         0.003907            1.849           0.064
    L4.Connection Type                0.007460         0.003445            2.165           0.030
    L4.Device                         0.059131         0.009763            6.057           0.000
    L4.Device Type                    0.002237         0.007980            0.280           0.779
    L4.Browser                       -0.015046         0.011690           -1.287           0.198
    L4.OS                             0.051768         0.036457            1.420           0.156
    L4.OS Version                    -0.001845         0.003811           -0.484           0.628
    L4.Device ID                      0.000022         0.000027            0.803           0.422
    L4.Happiness Score                0.000798         0.003247            0.246           0.806
    L4.Playback Stalls               -0.017380         0.056643           -0.307           0.759
    L4.Startup Error (Count)          0.069988         0.273062            0.256           0.798
    L4.Latency                       -0.000000         0.000000           -0.360           0.719
    L4.Crash Status                   0.200360         0.167434            1.197           0.231
    L4.End of Playback Status         0.041124         0.072224            0.569           0.569
    L4.User_ID_N                      0.000091         0.000086            1.061           0.289
    L4.Title_N                       -0.000001         0.000023           -0.047           0.962
    L4.Device_Vendor_N                0.005086         0.004826            1.054           0.292
    L4.Device_Model_N                -0.000303         0.000567           -0.534           0.593
    L4.Content_TV_Show_N              0.000039         0.000023            1.731           0.083
    L4.Country_N                      0.014758         0.007344            2.010           0.044
    L4.City_N                         0.000329         0.000138            2.389           0.017
    L4.Region_N                      -0.503867         0.136966           -3.679           0.000
    L5.Playtime                      -0.000004         0.000007           -0.649           0.516
    L5.Interruptions                 -0.000023         0.000860           -0.027           0.979
    L5.Join Time                      0.004423         0.003909            1.132           0.258
    L5.Buffer Ratio                   0.005192         0.003901            1.331           0.183
    L5.Connection Type                0.000763         0.003448            0.221           0.825
    L5.Device                         0.048073         0.009770            4.921           0.000
    L5.Device Type                   -0.002342         0.007990           -0.293           0.769
    L5.Browser                        0.017549         0.011708            1.499           0.134
    L5.OS                             0.111695         0.036493            3.061           0.002
    L5.OS Version                    -0.016463         0.003815           -4.316           0.000
    L5.Device ID                      0.000022         0.000027            0.835           0.404
    L5.Happiness Score                0.001355         0.003248            0.417           0.676
    L5.Playback Stalls                0.054142         0.055157            0.982           0.326
    L5.Startup Error (Count)         -0.063158         0.273037           -0.231           0.817
    L5.Latency                       -0.000000         0.000000           -0.601           0.548
    L5.Crash Status                   0.083905         0.167456            0.501           0.616
    L5.End of Playback Status         0.045888         0.072253            0.635           0.525
    L5.User_ID_N                     -0.000119         0.000086           -1.387           0.166
    L5.Title_N                       -0.000019         0.000023           -0.823           0.411
    L5.Device_Vendor_N               -0.002126         0.004829           -0.440           0.660
    L5.Device_Model_N                -0.001050         0.000567           -1.851           0.064
    L5.Content_TV_Show_N              0.000004         0.000023            0.194           0.846
    L5.Country_N                      0.008764         0.007357            1.191           0.234
    L5.City_N                         0.000205         0.000138            1.487           0.137
    L5.Region_N                      -0.205073         0.137152           -1.495           0.135
    L6.Playtime                       0.000000         0.000007            0.032           0.974
    L6.Interruptions                 -0.000489         0.000860           -0.569           0.569
    L6.Join Time                      0.000557         0.003911            0.142           0.887
    L6.Buffer Ratio                  -0.000445         0.003901           -0.114           0.909
    L6.Connection Type                0.004179         0.003445            1.213           0.225
    L6.Device                         0.047260         0.009764            4.840           0.000
    L6.Device Type                   -0.002215         0.007983           -0.277           0.781
    L6.Browser                       -0.037053         0.011690           -3.170           0.002
    L6.OS                             0.005049         0.036457            0.138           0.890
    L6.OS Version                     0.009007         0.003811            2.363           0.018
    L6.Device ID                      0.000012         0.000027            0.451           0.652
    L6.Happiness Score               -0.003065         0.003245           -0.944           0.345
    L6.Playback Stalls                0.117911         0.056522            2.086           0.037
    L6.Startup Error (Count)          0.028272         0.272998            0.104           0.918
    L6.Latency                        0.000000         0.000000            0.191           0.849
    L6.Crash Status                   0.009419         0.167444            0.056           0.955
    L6.End of Playback Status        -0.040164         0.072263           -0.556           0.578
    L6.User_ID_N                     -0.000009         0.000086           -0.103           0.918
    L6.Title_N                        0.000032         0.000023            1.414           0.157
    L6.Device_Vendor_N                0.013744         0.004825            2.848           0.004
    L6.Device_Model_N                -0.000651         0.000567           -1.149           0.251
    L6.Content_TV_Show_N             -0.000034         0.000023           -1.514           0.130
    L6.Country_N                      0.019114         0.007342            2.603           0.009
    L6.City_N                         0.000110         0.000138            0.801           0.423
    L6.Region_N                      -0.686785         0.136948           -5.015           0.000
    L7.Playtime                      -0.000005         0.000007           -0.813           0.416
    L7.Interruptions                  0.000006         0.000860            0.007           0.994
    L7.Join Time                     -0.000958         0.003910           -0.245           0.806
    L7.Buffer Ratio                  -0.003528         0.003902           -0.904           0.366
    L7.Connection Type                0.003107         0.003431            0.906           0.365
    L7.Device                         0.049244         0.009738            5.057           0.000
    L7.Device Type                   -0.020157         0.007952           -2.535           0.011
    L7.Browser                        0.025517         0.011652            2.190           0.029
    L7.OS                             0.018030         0.036355            0.496           0.620
    L7.OS Version                    -0.007962         0.003802           -2.094           0.036
    L7.Device ID                      0.000017         0.000027            0.631           0.528
    L7.Happiness Score               -0.004389         0.003241           -1.354           0.176
    L7.Playback Stalls                0.006283         0.056816            0.111           0.912
    L7.Startup Error (Count)          0.358898         0.272975            1.315           0.189
    L7.Latency                        0.000001         0.000000            1.988           0.047
    L7.Crash Status                  -0.376357         0.167441           -2.248           0.025
    L7.End of Playback Status        -0.009920         0.072253           -0.137           0.891
    L7.User_ID_N                     -0.000034         0.000086           -0.393           0.695
    L7.Title_N                        0.000006         0.000023            0.254           0.799
    L7.Device_Vendor_N                0.004157         0.004810            0.864           0.387
    L7.Device_Model_N                -0.000682         0.000565           -1.207           0.228
    L7.Content_TV_Show_N             -0.000023         0.000023           -0.995           0.320
    L7.Country_N                     -0.001766         0.007323           -0.241           0.809
    L7.City_N                        -0.000000         0.000137           -0.000           1.000
    L7.Region_N                      -0.206556         0.136572           -1.512           0.130
    L8.Playtime                       0.000012         0.000007            1.854           0.064
    L8.Interruptions                 -0.000328         0.000860           -0.381           0.703
    L8.Join Time                      0.007042         0.003910            1.801           0.072
    L8.Buffer Ratio                  -0.003578         0.003901           -0.917           0.359
    L8.Connection Type               -0.002709         0.003404           -0.796           0.426
    L8.Device                         0.012242         0.009688            1.264           0.206
    L8.Device Type                    0.009153         0.007892            1.160           0.246
    L8.Browser                       -0.019953         0.011590           -1.721           0.085
    L8.OS                             0.097422         0.036221            2.690           0.007
    L8.OS Version                    -0.007706         0.003790           -2.033           0.042
    L8.Device ID                      0.000028         0.000027            1.039           0.299
    L8.Happiness Score                0.001652         0.003232            0.511           0.609
    L8.Playback Stalls                0.019075         0.056817            0.336           0.737
    L8.Startup Error (Count)         -0.093601         0.272781           -0.343           0.731
    L8.Latency                       -0.000002         0.000000           -3.294           0.001
    L8.Crash Status                  -0.007070         0.167473           -0.042           0.966
    L8.End of Playback Status        -0.086040         0.072236           -1.191           0.234
    L8.User_ID_N                     -0.000005         0.000085           -0.060           0.952
    L8.Title_N                        0.000013         0.000023            0.560           0.576
    L8.Device_Vendor_N               -0.007530         0.004776           -1.577           0.115
    L8.Device_Model_N                -0.000193         0.000562           -0.344           0.731
    L8.Content_TV_Show_N             -0.000011         0.000023           -0.507           0.612
    L8.Country_N                      0.015477         0.007281            2.126           0.034
    L8.City_N                         0.000104         0.000136            0.762           0.446
    L8.Region_N                      -0.268648         0.136009           -1.975           0.048
    L9.Playtime                      -0.000004         0.000007           -0.533           0.594
    L9.Interruptions                 -0.000150         0.000860           -0.175           0.861
    L9.Join Time                      0.000631         0.003907            0.161           0.872
    L9.Buffer Ratio                  -0.004082         0.003901           -1.047           0.295
    L9.Connection Type               -0.004003         0.003361           -1.191           0.234
    L9.Device                         0.043305         0.009591            4.515           0.000
    L9.Device Type                    0.001799         0.007782            0.231           0.817
    L9.Browser                       -0.011847         0.011423           -1.037           0.300
    L9.OS                             0.039290         0.035798            1.098           0.272
    L9.OS Version                    -0.004374         0.003764           -1.162           0.245
    L9.Device ID                      0.000029         0.000026            1.094           0.274
    L9.Happiness Score               -0.002031         0.003188           -0.637           0.524
    L9.Playback Stalls               -0.086684         0.056620           -1.531           0.126
    L9.Startup Error (Count)         -0.565029         0.272535           -2.073           0.038
    L9.Latency                        0.000000         0.000000            0.120           0.905
    L9.Crash Status                   0.239011         0.167333            1.428           0.153
    L9.End of Playback Status        -0.082144         0.072210           -1.138           0.255
    L9.User_ID_N                     -0.000099         0.000084           -1.187           0.235
    L9.Title_N                        0.000012         0.000023            0.520           0.603
    L9.Device_Vendor_N               -0.001369         0.004714           -0.290           0.772
    L9.Device_Model_N                -0.001382         0.000556           -2.487           0.013
    L9.Content_TV_Show_N             -0.000015         0.000022           -0.674           0.501
    L9.Country_N                      0.026406         0.007230            3.652           0.000
    L9.City_N                         0.000035         0.000134            0.263           0.793
    L9.Region_N                      -0.376004         0.135138           -2.782           0.005
    ============================================================================================
    
    Results for equation Device Type
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             3.702976         0.372200            9.949           0.000
    L1.Playtime                       0.000003         0.000005            0.741           0.459
    L1.Interruptions                 -0.000434         0.000582           -0.745           0.456
    L1.Join Time                      0.007037         0.002659            2.647           0.008
    L1.Buffer Ratio                  -0.003167         0.002638           -1.200           0.230
    L1.Connection Type               -0.013494         0.002276           -5.930           0.000
    L1.Device                        -0.009323         0.006498           -1.435           0.151
    L1.Device Type                    0.166104         0.005271           31.510           0.000
    L1.Browser                       -0.027632         0.007733           -3.573           0.000
    L1.OS                             0.069784         0.024263            2.876           0.004
    L1.OS Version                    -0.006395         0.002549           -2.509           0.012
    L1.Device ID                      0.000034         0.000018            1.893           0.058
    L1.Happiness Score               -0.000031         0.002161           -0.014           0.989
    L1.Playback Stalls                0.036481         0.038360            0.951           0.342
    L1.Startup Error (Count)         -0.437576         0.184774           -2.368           0.018
    L1.Latency                       -0.000000         0.000000           -0.923           0.356
    L1.Crash Status                   0.414561         0.113112            3.665           0.000
    L1.End of Playback Status        -0.058641         0.048731           -1.203           0.229
    L1.User_ID_N                     -0.000105         0.000057           -1.855           0.064
    L1.Title_N                        0.000037         0.000015            2.402           0.016
    L1.Device_Vendor_N               -0.008990         0.003192           -2.817           0.005
    L1.Device_Model_N                -0.000777         0.000376           -2.064           0.039
    L1.Content_TV_Show_N              0.000034         0.000015            2.240           0.025
    L1.Country_N                      0.024441         0.004896            4.992           0.000
    L1.City_N                         0.000505         0.000091            5.557           0.000
    L1.Region_N                      -0.066898         0.091469           -0.731           0.465
    L2.Playtime                      -0.000002         0.000005           -0.532           0.595
    L2.Interruptions                  0.000763         0.000582            1.310           0.190
    L2.Join Time                      0.002602         0.002660            0.978           0.328
    L2.Buffer Ratio                  -0.004004         0.002644           -1.514           0.130
    L2.Connection Type                0.000481         0.002304            0.209           0.835
    L2.Device                        -0.010004         0.006561           -1.525           0.127
    L2.Device Type                    0.138594         0.005343           25.940           0.000
    L2.Browser                       -0.011337         0.007844           -1.445           0.148
    L2.OS                             0.034610         0.024528            1.411           0.158
    L2.OS Version                    -0.001049         0.002567           -0.409           0.683
    L2.Device ID                     -0.000004         0.000018           -0.213           0.831
    L2.Happiness Score                0.003094         0.002190            1.413           0.158
    L2.Playback Stalls               -0.032836         0.038363           -0.856           0.392
    L2.Startup Error (Count)         -0.460639         0.184880           -2.492           0.013
    L2.Latency                       -0.000000         0.000000           -0.364           0.716
    L2.Crash Status                   0.202128         0.113393            1.783           0.075
    L2.End of Playback Status        -0.223547         0.048838           -4.577           0.000
    L2.User_ID_N                     -0.000019         0.000058           -0.332           0.740
    L2.Title_N                        0.000018         0.000015            1.186           0.236
    L2.Device_Vendor_N               -0.000312         0.003233           -0.097           0.923
    L2.Device_Model_N                -0.000247         0.000381           -0.649           0.516
    L2.Content_TV_Show_N             -0.000030         0.000015           -1.964           0.050
    L2.Country_N                      0.003930         0.004931            0.797           0.425
    L2.City_N                         0.000254         0.000092            2.755           0.006
    L2.Region_N                       0.084885         0.092093            0.922           0.357
    L3.Playtime                       0.000016         0.000005            3.620           0.000
    L3.Interruptions                  0.000004         0.000582            0.006           0.995
    L3.Join Time                     -0.004101         0.002660           -1.542           0.123
    L3.Buffer Ratio                  -0.004989         0.002645           -1.886           0.059
    L3.Connection Type               -0.004758         0.002323           -2.048           0.041
    L3.Device                        -0.012399         0.006595           -1.880           0.060
    L3.Device Type                    0.107967         0.005384           20.054           0.000
    L3.Browser                        0.010129         0.007888            1.284           0.199
    L3.OS                             0.031333         0.024616            1.273           0.203
    L3.OS Version                    -0.004629         0.002573           -1.799           0.072
    L3.Device ID                      0.000013         0.000018            0.727           0.467
    L3.Happiness Score                0.000020         0.002196            0.009           0.993
    L3.Playback Stalls               -0.046004         0.038365           -1.199           0.230
    L3.Startup Error (Count)         -0.407897         0.184884           -2.206           0.027
    L3.Latency                        0.000000         0.000000            0.386           0.700
    L3.Crash Status                   0.253352         0.113360            2.235           0.025
    L3.End of Playback Status        -0.099428         0.048888           -2.034           0.042
    L3.User_ID_N                     -0.000102         0.000058           -1.753           0.080
    L3.Title_N                       -0.000014         0.000015           -0.935           0.350
    L3.Device_Vendor_N               -0.000216         0.003256           -0.066           0.947
    L3.Device_Model_N                 0.000577         0.000383            1.508           0.132
    L3.Content_TV_Show_N             -0.000023         0.000015           -1.528           0.127
    L3.Country_N                     -0.006633         0.004959           -1.337           0.181
    L3.City_N                        -0.000100         0.000093           -1.076           0.282
    L3.Region_N                       0.228347         0.092470            2.469           0.014
    L4.Playtime                      -0.000004         0.000005           -0.943           0.345
    L4.Interruptions                 -0.000487         0.000582           -0.837           0.403
    L4.Join Time                      0.002674         0.002646            1.011           0.312
    L4.Buffer Ratio                   0.000522         0.002645            0.197           0.843
    L4.Connection Type               -0.000923         0.002332           -0.396           0.692
    L4.Device                         0.014963         0.006610            2.264           0.024
    L4.Device Type                    0.071405         0.005403           13.216           0.000
    L4.Browser                        0.004457         0.007915            0.563           0.573
    L4.OS                            -0.034421         0.024683           -1.395           0.163
    L4.OS Version                    -0.000171         0.002580           -0.066           0.947
    L4.Device ID                     -0.000007         0.000018           -0.387           0.699
    L4.Happiness Score                0.000014         0.002198            0.006           0.995
    L4.Playback Stalls                0.004682         0.038350            0.122           0.903
    L4.Startup Error (Count)         -0.007794         0.184874           -0.042           0.966
    L4.Latency                        0.000000         0.000000            0.203           0.839
    L4.Crash Status                  -0.057505         0.113359           -0.507           0.612
    L4.End of Playback Status         0.014362         0.048899            0.294           0.769
    L4.User_ID_N                     -0.000040         0.000058           -0.694           0.488
    L4.Title_N                        0.000010         0.000015            0.632           0.528
    L4.Device_Vendor_N               -0.003342         0.003267           -1.023           0.306
    L4.Device_Model_N                -0.000259         0.000384           -0.675           0.500
    L4.Content_TV_Show_N             -0.000010         0.000015           -0.659           0.510
    L4.Country_N                     -0.006110         0.004972           -1.229           0.219
    L4.City_N                        -0.000004         0.000093           -0.048           0.962
    L4.Region_N                       0.072718         0.092732            0.784           0.433
    L5.Playtime                       0.000011         0.000005            2.412           0.016
    L5.Interruptions                  0.000196         0.000582            0.336           0.737
    L5.Join Time                     -0.002913         0.002647           -1.101           0.271
    L5.Buffer Ratio                  -0.001221         0.002641           -0.462           0.644
    L5.Connection Type                0.004565         0.002334            1.956           0.051
    L5.Device                        -0.006261         0.006615           -0.947           0.344
    L5.Device Type                    0.051374         0.005410            9.497           0.000
    L5.Browser                        0.009554         0.007927            1.205           0.228
    L5.OS                            -0.033205         0.024707           -1.344           0.179
    L5.OS Version                     0.003778         0.002583            1.463           0.143
    L5.Device ID                     -0.000010         0.000018           -0.564           0.572
    L5.Happiness Score                0.002460         0.002199            1.119           0.263
    L5.Playback Stalls               -0.086340         0.037343           -2.312           0.021
    L5.Startup Error (Count)          0.142745         0.184857            0.772           0.440
    L5.Latency                       -0.000000         0.000000           -1.057           0.291
    L5.Crash Status                  -0.167415         0.113375           -1.477           0.140
    L5.End of Playback Status        -0.025559         0.048918           -0.522           0.601
    L5.User_ID_N                     -0.000012         0.000058           -0.200           0.842
    L5.Title_N                       -0.000013         0.000015           -0.845           0.398
    L5.Device_Vendor_N                0.002922         0.003270            0.894           0.371
    L5.Device_Model_N                 0.001079         0.000384            2.810           0.005
    L5.Content_TV_Show_N             -0.000007         0.000015           -0.435           0.664
    L5.Country_N                     -0.005501         0.004981           -1.104           0.269
    L5.City_N                        -0.000027         0.000093           -0.284           0.776
    L5.Region_N                       0.127503         0.092858            1.373           0.170
    L6.Playtime                       0.000008         0.000005            1.828           0.068
    L6.Interruptions                 -0.000908         0.000582           -1.559           0.119
    L6.Join Time                      0.000532         0.002648            0.201           0.841
    L6.Buffer Ratio                  -0.003893         0.002641           -1.474           0.140
    L6.Connection Type                0.002433         0.002332            1.043           0.297
    L6.Device                        -0.006490         0.006611           -0.982           0.326
    L6.Device Type                    0.048515         0.005405            8.977           0.000
    L6.Browser                        0.019745         0.007914            2.495           0.013
    L6.OS                             0.003441         0.024683            0.139           0.889
    L6.OS Version                    -0.003920         0.002580           -1.519           0.129
    L6.Device ID                     -0.000011         0.000018           -0.593           0.553
    L6.Happiness Score               -0.001701         0.002197           -0.774           0.439
    L6.Playback Stalls               -0.005928         0.038268           -0.155           0.877
    L6.Startup Error (Count)          0.238960         0.184831            1.293           0.196
    L6.Latency                        0.000001         0.000000            2.034           0.042
    L6.Crash Status                  -0.238124         0.113367           -2.100           0.036
    L6.End of Playback Status         0.091513         0.048925            1.870           0.061
    L6.User_ID_N                      0.000032         0.000058            0.542           0.588
    L6.Title_N                       -0.000049         0.000015           -3.157           0.002
    L6.Device_Vendor_N                0.002950         0.003267            0.903           0.366
    L6.Device_Model_N                 0.000579         0.000384            1.509           0.131
    L6.Content_TV_Show_N              0.000011         0.000015            0.744           0.457
    L6.Country_N                     -0.017325         0.004971           -3.485           0.000
    L6.City_N                        -0.000035         0.000093           -0.371           0.711
    L6.Region_N                       0.394595         0.092720            4.256           0.000
    L7.Playtime                       0.000011         0.000005            2.374           0.018
    L7.Interruptions                  0.000355         0.000582            0.609           0.543
    L7.Join Time                      0.003209         0.002647            1.212           0.225
    L7.Buffer Ratio                   0.000329         0.002642            0.125           0.901
    L7.Connection Type               -0.001563         0.002323           -0.673           0.501
    L7.Device                        -0.001394         0.006593           -0.212           0.832
    L7.Device Type                    0.035920         0.005384            6.672           0.000
    L7.Browser                        0.012611         0.007889            1.599           0.110
    L7.OS                            -0.059318         0.024614           -2.410           0.016
    L7.OS Version                     0.004235         0.002574            1.646           0.100
    L7.Device ID                     -0.000029         0.000018           -1.621           0.105
    L7.Happiness Score               -0.000270         0.002194           -0.123           0.902
    L7.Playback Stalls               -0.046441         0.038467           -1.207           0.227
    L7.Startup Error (Count)         -0.040785         0.184815           -0.221           0.825
    L7.Latency                       -0.000001         0.000000           -2.995           0.003
    L7.Crash Status                  -0.018589         0.113364           -0.164           0.870
    L7.End of Playback Status         0.049348         0.048918            1.009           0.313
    L7.User_ID_N                     -0.000061         0.000058           -1.056           0.291
    L7.Title_N                       -0.000030         0.000015           -1.939           0.052
    L7.Device_Vendor_N               -0.001338         0.003256           -0.411           0.681
    L7.Device_Model_N                 0.000948         0.000383            2.477           0.013
    L7.Content_TV_Show_N              0.000015         0.000015            0.989           0.323
    L7.Country_N                     -0.011108         0.004958           -2.241           0.025
    L7.City_N                        -0.000152         0.000093           -1.637           0.102
    L7.Region_N                       0.120555         0.092465            1.304           0.192
    L8.Playtime                      -0.000001         0.000005           -0.232           0.817
    L8.Interruptions                  0.000318         0.000582            0.546           0.585
    L8.Join Time                     -0.006498         0.002647           -2.455           0.014
    L8.Buffer Ratio                   0.002179         0.002641            0.825           0.409
    L8.Connection Type                0.002684         0.002305            1.165           0.244
    L8.Device                        -0.012146         0.006559           -1.852           0.064
    L8.Device Type                    0.028677         0.005344            5.367           0.000
    L8.Browser                        0.014883         0.007847            1.897           0.058
    L8.OS                            -0.013111         0.024523           -0.535           0.593
    L8.OS Version                     0.000372         0.002566            0.145           0.885
    L8.Device ID                     -0.000046         0.000018           -2.539           0.011
    L8.Happiness Score               -0.001790         0.002188           -0.818           0.413
    L8.Playback Stalls               -0.047006         0.038468           -1.222           0.222
    L8.Startup Error (Count)         -0.050952         0.184684           -0.276           0.783
    L8.Latency                        0.000000         0.000000            1.357           0.175
    L8.Crash Status                  -0.048782         0.113386           -0.430           0.667
    L8.End of Playback Status        -0.010784         0.048906           -0.221           0.825
    L8.User_ID_N                      0.000022         0.000058            0.376           0.707
    L8.Title_N                       -0.000018         0.000015           -1.167           0.243
    L8.Device_Vendor_N                0.002010         0.003234            0.621           0.534
    L8.Device_Model_N                 0.000492         0.000381            1.291           0.197
    L8.Content_TV_Show_N              0.000010         0.000015            0.669           0.503
    L8.Country_N                     -0.017543         0.004930           -3.559           0.000
    L8.City_N                        -0.000161         0.000092           -1.750           0.080
    L8.Region_N                       0.325438         0.092084            3.534           0.000
    L9.Playtime                      -0.000000         0.000005           -0.091           0.927
    L9.Interruptions                  0.000361         0.000582            0.619           0.536
    L9.Join Time                     -0.001589         0.002646           -0.601           0.548
    L9.Buffer Ratio                  -0.002110         0.002641           -0.799           0.424
    L9.Connection Type                0.002669         0.002276            1.173           0.241
    L9.Device                         0.008849         0.006494            1.363           0.173
    L9.Device Type                    0.043360         0.005269            8.230           0.000
    L9.Browser                       -0.002493         0.007734           -0.322           0.747
    L9.OS                            -0.007408         0.024237           -0.306           0.760
    L9.OS Version                     0.001171         0.002548            0.459           0.646
    L9.Device ID                     -0.000014         0.000018           -0.768           0.443
    L9.Happiness Score                0.001490         0.002159            0.690           0.490
    L9.Playback Stalls                0.085023         0.038334            2.218           0.027
    L9.Startup Error (Count)          0.193993         0.184518            1.051           0.293
    L9.Latency                       -0.000000         0.000000           -0.232           0.817
    L9.Crash Status                  -0.193814         0.113291           -1.711           0.087
    L9.End of Playback Status        -0.004435         0.048889           -0.091           0.928
    L9.User_ID_N                     -0.000010         0.000057           -0.172           0.864
    L9.Title_N                       -0.000033         0.000015           -2.143           0.032
    L9.Device_Vendor_N                0.003853         0.003192            1.207           0.227
    L9.Device_Model_N                 0.000166         0.000376            0.440           0.660
    L9.Content_TV_Show_N              0.000014         0.000015            0.906           0.365
    L9.Country_N                     -0.016360         0.004895           -3.342           0.001
    L9.City_N                        -0.000037         0.000091           -0.408           0.683
    L9.Region_N                       0.179323         0.091494            1.960           0.050
    ============================================================================================
    
    Results for equation Browser
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             0.331810         0.365368            0.908           0.364
    L1.Playtime                       0.000003         0.000004            0.694           0.488
    L1.Interruptions                 -0.000241         0.000572           -0.422           0.673
    L1.Join Time                     -0.001888         0.002610           -0.724           0.469
    L1.Buffer Ratio                  -0.003130         0.002590           -1.209           0.227
    L1.Connection Type                0.000508         0.002234            0.227           0.820
    L1.Device                         0.042945         0.006379            6.733           0.000
    L1.Device Type                   -0.011206         0.005175           -2.166           0.030
    L1.Browser                        0.162145         0.007591           21.360           0.000
    L1.OS                            -0.141521         0.023818           -5.942           0.000
    L1.OS Version                     0.002709         0.002502            1.083           0.279
    L1.Device ID                      0.000008         0.000017            0.438           0.661
    L1.Happiness Score                0.002527         0.002121            1.191           0.233
    L1.Playback Stalls               -0.018159         0.037656           -0.482           0.630
    L1.Startup Error (Count)          0.444037         0.181383            2.448           0.014
    L1.Latency                        0.000000         0.000000            0.986           0.324
    L1.Crash Status                  -0.264213         0.111036           -2.380           0.017
    L1.End of Playback Status         0.076358         0.047836            1.596           0.110
    L1.User_ID_N                      0.000131         0.000056            2.362           0.018
    L1.Title_N                       -0.000012         0.000015           -0.766           0.443
    L1.Device_Vendor_N               -0.006520         0.003133           -2.081           0.037
    L1.Device_Model_N                 0.001118         0.000369            3.026           0.002
    L1.Content_TV_Show_N             -0.000083         0.000015           -5.541           0.000
    L1.Country_N                     -0.010814         0.004806           -2.250           0.024
    L1.City_N                        -0.000336         0.000089           -3.765           0.000
    L1.Region_N                       0.063229         0.089790            0.704           0.481
    L2.Playtime                      -0.000007         0.000004           -1.514           0.130
    L2.Interruptions                 -0.000031         0.000572           -0.054           0.957
    L2.Join Time                     -0.005480         0.002611           -2.099           0.036
    L2.Buffer Ratio                  -0.000414         0.002596           -0.160           0.873
    L2.Connection Type                0.000890         0.002262            0.393           0.694
    L2.Device                         0.009644         0.006441            1.497           0.134
    L2.Device Type                   -0.009263         0.005245           -1.766           0.077
    L2.Browser                        0.120851         0.007700           15.695           0.000
    L2.OS                            -0.029135         0.024078           -1.210           0.226
    L2.OS Version                     0.001988         0.002519            0.789           0.430
    L2.Device ID                      0.000003         0.000018            0.166           0.868
    L2.Happiness Score                0.005248         0.002149            2.442           0.015
    L2.Playback Stalls                0.017303         0.037659            0.459           0.646
    L2.Startup Error (Count)          0.547196         0.181487            3.015           0.003
    L2.Latency                        0.000000         0.000000            0.092           0.926
    L2.Crash Status                  -0.326195         0.111312           -2.930           0.003
    L2.End of Playback Status         0.143499         0.047941            2.993           0.003
    L2.User_ID_N                      0.000120         0.000056            2.127           0.033
    L2.Title_N                        0.000005         0.000015            0.335           0.738
    L2.Device_Vendor_N                0.001179         0.003174            0.371           0.710
    L2.Device_Model_N                 0.000548         0.000374            1.467           0.142
    L2.Content_TV_Show_N              0.000001         0.000015            0.068           0.946
    L2.Country_N                      0.002122         0.004841            0.438           0.661
    L2.City_N                        -0.000185         0.000091           -2.038           0.042
    L2.Region_N                      -0.176218         0.090403           -1.949           0.051
    L3.Playtime                      -0.000000         0.000004           -0.089           0.929
    L3.Interruptions                 -0.000114         0.000572           -0.200           0.842
    L3.Join Time                      0.006043         0.002611            2.315           0.021
    L3.Buffer Ratio                  -0.003549         0.002597           -1.367           0.172
    L3.Connection Type                0.002938         0.002281            1.288           0.198
    L3.Device                         0.013592         0.006474            2.100           0.036
    L3.Device Type                   -0.011556         0.005285           -2.187           0.029
    L3.Browser                        0.090496         0.007743           11.687           0.000
    L3.OS                            -0.012457         0.024164           -0.516           0.606
    L3.OS Version                    -0.002846         0.002526           -1.127           0.260
    L3.Device ID                      0.000003         0.000018            0.143           0.886
    L3.Happiness Score                0.002976         0.002156            1.381           0.167
    L3.Playback Stalls                0.006633         0.037661            0.176           0.860
    L3.Startup Error (Count)          0.337481         0.181490            1.859           0.063
    L3.Latency                       -0.000000         0.000000           -0.753           0.452
    L3.Crash Status                  -0.108768         0.111279           -0.977           0.328
    L3.End of Playback Status         0.086323         0.047991            1.799           0.072
    L3.User_ID_N                      0.000045         0.000057            0.795           0.426
    L3.Title_N                        0.000008         0.000015            0.530           0.596
    L3.Device_Vendor_N                0.001784         0.003196            0.558           0.577
    L3.Device_Model_N                -0.000261         0.000376           -0.695           0.487
    L3.Content_TV_Show_N              0.000014         0.000015            0.944           0.345
    L3.Country_N                      0.008496         0.004868            1.745           0.081
    L3.City_N                         0.000062         0.000091            0.683           0.495
    L3.Region_N                      -0.203823         0.090773           -2.245           0.025
    L4.Playtime                      -0.000002         0.000004           -0.356           0.721
    L4.Interruptions                 -0.000126         0.000572           -0.221           0.825
    L4.Join Time                      0.000066         0.002597            0.025           0.980
    L4.Buffer Ratio                   0.005331         0.002597            2.053           0.040
    L4.Connection Type                0.005762         0.002290            2.516           0.012
    L4.Device                         0.000626         0.006489            0.096           0.923
    L4.Device Type                   -0.001744         0.005304           -0.329           0.742
    L4.Browser                        0.064286         0.007770            8.274           0.000
    L4.OS                             0.044162         0.024230            1.823           0.068
    L4.OS Version                    -0.004516         0.002533           -1.783           0.075
    L4.Device ID                      0.000018         0.000018            1.002           0.316
    L4.Happiness Score               -0.001079         0.002158           -0.500           0.617
    L4.Playback Stalls               -0.023472         0.037646           -0.623           0.533
    L4.Startup Error (Count)         -0.098279         0.181481           -0.542           0.588
    L4.Latency                       -0.000000         0.000000           -0.685           0.493
    L4.Crash Status                   0.176425         0.111279            1.585           0.113
    L4.End of Playback Status         0.004555         0.048001            0.095           0.924
    L4.User_ID_N                     -0.000026         0.000057           -0.463           0.643
    L4.Title_N                       -0.000006         0.000015           -0.398           0.691
    L4.Device_Vendor_N                0.005210         0.003207            1.625           0.104
    L4.Device_Model_N                -0.000623         0.000377           -1.654           0.098
    L4.Content_TV_Show_N              0.000024         0.000015            1.580           0.114
    L4.Country_N                     -0.000657         0.004881           -0.135           0.893
    L4.City_N                         0.000203         0.000092            2.214           0.027
    L4.Region_N                      -0.102228         0.091030           -1.123           0.261
    L5.Playtime                      -0.000003         0.000004           -0.626           0.531
    L5.Interruptions                 -0.000016         0.000572           -0.028           0.978
    L5.Join Time                      0.006276         0.002598            2.416           0.016
    L5.Buffer Ratio                   0.004590         0.002593            1.770           0.077
    L5.Connection Type                0.000452         0.002291            0.197           0.844
    L5.Device                        -0.008906         0.006493           -1.372           0.170
    L5.Device Type                    0.005397         0.005310            1.016           0.310
    L5.Browser                        0.042179         0.007782            5.420           0.000
    L5.OS                             0.095977         0.024254            3.957           0.000
    L5.OS Version                    -0.007842         0.002535           -3.093           0.002
    L5.Device ID                      0.000009         0.000018            0.479           0.632
    L5.Happiness Score                0.004256         0.002158            1.972           0.049
    L5.Playback Stalls               -0.016814         0.036658           -0.459           0.646
    L5.Startup Error (Count)          0.080405         0.181464            0.443           0.658
    L5.Latency                       -0.000000         0.000000           -0.231           0.817
    L5.Crash Status                  -0.006951         0.111294           -0.062           0.950
    L5.End of Playback Status         0.048742         0.048020            1.015           0.310
    L5.User_ID_N                     -0.000083         0.000057           -1.451           0.147
    L5.Title_N                       -0.000022         0.000015           -1.436           0.151
    L5.Device_Vendor_N               -0.000234         0.003210           -0.073           0.942
    L5.Device_Model_N                -0.000514         0.000377           -1.364           0.172
    L5.Content_TV_Show_N              0.000012         0.000015            0.769           0.442
    L5.Country_N                      0.004686         0.004890            0.958           0.338
    L5.City_N                         0.000228         0.000092            2.486           0.013
    L5.Region_N                      -0.125357         0.091153           -1.375           0.169
    L6.Playtime                      -0.000001         0.000004           -0.319           0.750
    L6.Interruptions                 -0.000005         0.000572           -0.009           0.992
    L6.Join Time                      0.001915         0.002599            0.737           0.461
    L6.Buffer Ratio                  -0.001563         0.002593           -0.603           0.547
    L6.Connection Type                0.002689         0.002290            1.174           0.240
    L6.Device                         0.002873         0.006489            0.443           0.658
    L6.Device Type                    0.001317         0.005305            0.248           0.804
    L6.Browser                        0.020330         0.007769            2.617           0.009
    L6.OS                            -0.001427         0.024230           -0.059           0.953
    L6.OS Version                     0.004860         0.002533            1.919           0.055
    L6.Device ID                      0.000004         0.000018            0.218           0.827
    L6.Happiness Score               -0.001118         0.002157           -0.518           0.604
    L6.Playback Stalls                0.071275         0.037566            1.897           0.058
    L6.Startup Error (Count)          0.093861         0.181439            0.517           0.605
    L6.Latency                       -0.000000         0.000000           -0.584           0.559
    L6.Crash Status                  -0.003311         0.111286           -0.030           0.976
    L6.End of Playback Status        -0.001388         0.048027           -0.029           0.977
    L6.User_ID_N                      0.000020         0.000057            0.342           0.733
    L6.Title_N                       -0.000001         0.000015           -0.093           0.926
    L6.Device_Vendor_N                0.007891         0.003207            2.461           0.014
    L6.Device_Model_N                -0.000205         0.000377           -0.544           0.587
    L6.Content_TV_Show_N             -0.000025         0.000015           -1.686           0.092
    L6.Country_N                      0.018417         0.004880            3.774           0.000
    L6.City_N                         0.000084         0.000092            0.915           0.360
    L6.Region_N                      -0.404600         0.091018           -4.445           0.000
    L7.Playtime                       0.000004         0.000004            0.797           0.425
    L7.Interruptions                 -0.000180         0.000572           -0.314           0.753
    L7.Join Time                     -0.001128         0.002599           -0.434           0.664
    L7.Buffer Ratio                  -0.003584         0.002593           -1.382           0.167
    L7.Connection Type                0.002546         0.002280            1.117           0.264
    L7.Device                         0.008707         0.006472            1.345           0.179
    L7.Device Type                   -0.010197         0.005285           -1.929           0.054
    L7.Browser                        0.047776         0.007744            6.169           0.000
    L7.OS                             0.017418         0.024162            0.721           0.471
    L7.OS Version                    -0.006208         0.002527           -2.457           0.014
    L7.Device ID                      0.000018         0.000018            1.029           0.304
    L7.Happiness Score               -0.003095         0.002154           -1.437           0.151
    L7.Playback Stalls               -0.012920         0.037761           -0.342           0.732
    L7.Startup Error (Count)          0.161951         0.181423            0.893           0.372
    L7.Latency                        0.000000         0.000000            0.177           0.859
    L7.Crash Status                  -0.251754         0.111284           -2.262           0.024
    L7.End of Playback Status        -0.029924         0.048021           -0.623           0.533
    L7.User_ID_N                     -0.000095         0.000057           -1.662           0.096
    L7.Title_N                       -0.000010         0.000015           -0.634           0.526
    L7.Device_Vendor_N                0.002308         0.003197            0.722           0.470
    L7.Device_Model_N                -0.000671         0.000376           -1.785           0.074
    L7.Content_TV_Show_N             -0.000009         0.000015           -0.573           0.566
    L7.Country_N                      0.003334         0.004867            0.685           0.493
    L7.City_N                         0.000042         0.000091            0.463           0.643
    L7.Region_N                      -0.159991         0.090768           -1.763           0.078
    L8.Playtime                       0.000004         0.000004            0.990           0.322
    L8.Interruptions                 -0.000211         0.000572           -0.369           0.712
    L8.Join Time                      0.001545         0.002598            0.595           0.552
    L8.Buffer Ratio                  -0.001507         0.002593           -0.581           0.561
    L8.Connection Type               -0.004043         0.002262           -1.787           0.074
    L8.Device                        -0.016022         0.006439           -2.488           0.013
    L8.Device Type                    0.010476         0.005245            1.997           0.046
    L8.Browser                        0.015790         0.007703            2.050           0.040
    L8.OS                             0.076766         0.024073            3.189           0.001
    L8.OS Version                    -0.004672         0.002519           -1.855           0.064
    L8.Device ID                      0.000027         0.000018            1.537           0.124
    L8.Happiness Score                0.001211         0.002148            0.564           0.573
    L8.Playback Stalls               -0.015578         0.037762           -0.413           0.680
    L8.Startup Error (Count)          0.060387         0.181294            0.333           0.739
    L8.Latency                       -0.000001         0.000000           -2.195           0.028
    L8.Crash Status                  -0.079068         0.111305           -0.710           0.477
    L8.End of Playback Status        -0.065724         0.048009           -1.369           0.171
    L8.User_ID_N                     -0.000007         0.000056           -0.121           0.903
    L8.Title_N                        0.000008         0.000015            0.509           0.610
    L8.Device_Vendor_N               -0.005493         0.003174           -1.731           0.084
    L8.Device_Model_N                -0.000716         0.000374           -1.917           0.055
    L8.Content_TV_Show_N              0.000009         0.000015            0.583           0.560
    L8.Country_N                      0.010249         0.004839            2.118           0.034
    L8.City_N                        -0.000066         0.000091           -0.734           0.463
    L8.Region_N                      -0.080924         0.090394           -0.895           0.371
    L9.Playtime                      -0.000003         0.000004           -0.677           0.499
    L9.Interruptions                 -0.000033         0.000572           -0.058           0.954
    L9.Join Time                     -0.000359         0.002597           -0.138           0.890
    L9.Buffer Ratio                   0.001553         0.002592            0.599           0.549
    L9.Connection Type               -0.001058         0.002234           -0.474           0.636
    L9.Device                         0.001361         0.006375            0.213           0.831
    L9.Device Type                    0.006772         0.005172            1.309           0.190
    L9.Browser                        0.033898         0.007592            4.465           0.000
    L9.OS                             0.046156         0.023792            1.940           0.052
    L9.OS Version                    -0.004064         0.002501           -1.625           0.104
    L9.Device ID                      0.000008         0.000018            0.480           0.631
    L9.Happiness Score               -0.001408         0.002119           -0.665           0.506
    L9.Playback Stalls               -0.043928         0.037631           -1.167           0.243
    L9.Startup Error (Count)         -0.388185         0.181131           -2.143           0.032
    L9.Latency                       -0.000000         0.000000           -1.480           0.139
    L9.Crash Status                   0.112305         0.111212            1.010           0.313
    L9.End of Playback Status        -0.074914         0.047992           -1.561           0.119
    L9.User_ID_N                     -0.000006         0.000056           -0.106           0.916
    L9.Title_N                       -0.000001         0.000015           -0.075           0.941
    L9.Device_Vendor_N                0.006416         0.003133            2.048           0.041
    L9.Device_Model_N                -0.000879         0.000369           -2.381           0.017
    L9.Content_TV_Show_N             -0.000014         0.000015           -0.941           0.347
    L9.Country_N                      0.019134         0.004805            3.982           0.000
    L9.City_N                        -0.000073         0.000089           -0.823           0.410
    L9.Region_N                      -0.153751         0.089815           -1.712           0.087
    ============================================================================================
    
    Results for equation OS
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             0.169502         0.298553            0.568           0.570
    L1.Playtime                       0.000000         0.000004            0.006           0.995
    L1.Interruptions                 -0.000418         0.000467           -0.896           0.370
    L1.Join Time                     -0.002354         0.002133           -1.104           0.270
    L1.Buffer Ratio                   0.001894         0.002116            0.895           0.371
    L1.Connection Type                0.003810         0.001825            2.087           0.037
    L1.Device                         0.025592         0.005212            4.910           0.000
    L1.Device Type                   -0.007411         0.004228           -1.753           0.080
    L1.Browser                        0.000653         0.006203            0.105           0.916
    L1.OS                             0.084482         0.019462            4.341           0.000
    L1.OS Version                     0.002617         0.002045            1.280           0.200
    L1.Device ID                     -0.000015         0.000014           -1.070           0.285
    L1.Happiness Score                0.003007         0.001733            1.735           0.083
    L1.Playback Stalls               -0.039995         0.030770           -1.300           0.194
    L1.Startup Error (Count)          0.485485         0.148213            3.276           0.001
    L1.Latency                        0.000000         0.000000            0.906           0.365
    L1.Crash Status                  -0.407258         0.090730           -4.489           0.000
    L1.End of Playback Status         0.102230         0.039088            2.615           0.009
    L1.User_ID_N                      0.000129         0.000045            2.845           0.004
    L1.Title_N                       -0.000004         0.000012           -0.328           0.743
    L1.Device_Vendor_N                0.006440         0.002560            2.515           0.012
    L1.Device_Model_N                 0.001063         0.000302            3.522           0.000
    L1.Content_TV_Show_N             -0.000073         0.000012           -5.947           0.000
    L1.Country_N                     -0.016822         0.003927           -4.283           0.000
    L1.City_N                        -0.000338         0.000073           -4.632           0.000
    L1.Region_N                       0.148952         0.073370            2.030           0.042
    L2.Playtime                      -0.000004         0.000004           -1.083           0.279
    L2.Interruptions                 -0.000214         0.000467           -0.459           0.646
    L2.Join Time                     -0.003316         0.002133           -1.554           0.120
    L2.Buffer Ratio                   0.001280         0.002121            0.604           0.546
    L2.Connection Type               -0.000542         0.001848           -0.293           0.769
    L2.Device                         0.008068         0.005263            1.533           0.125
    L2.Device Type                   -0.009600         0.004286           -2.240           0.025
    L2.Browser                        0.007324         0.006292            1.164           0.244
    L2.OS                             0.086753         0.019675            4.409           0.000
    L2.OS Version                     0.000408         0.002059            0.198           0.843
    L2.Device ID                     -0.000004         0.000014           -0.245           0.807
    L2.Happiness Score                0.002777         0.001756            1.581           0.114
    L2.Playback Stalls                0.042539         0.030772            1.382           0.167
    L2.Startup Error (Count)          0.362069         0.148298            2.441           0.015
    L2.Latency                        0.000000         0.000000            0.535           0.593
    L2.Crash Status                  -0.205790         0.090956           -2.263           0.024
    L2.End of Playback Status         0.133777         0.039174            3.415           0.001
    L2.User_ID_N                      0.000051         0.000046            1.111           0.267
    L2.Title_N                        0.000005         0.000012            0.365           0.715
    L2.Device_Vendor_N               -0.001010         0.002594           -0.390           0.697
    L2.Device_Model_N                 0.000601         0.000305            1.967           0.049
    L2.Content_TV_Show_N             -0.000008         0.000012           -0.670           0.503
    L2.Country_N                      0.002341         0.003955            0.592           0.554
    L2.City_N                        -0.000180         0.000074           -2.438           0.015
    L2.Region_N                      -0.162663         0.073871           -2.202           0.028
    L3.Playtime                      -0.000004         0.000004           -1.083           0.279
    L3.Interruptions                 -0.000139         0.000467           -0.298           0.766
    L3.Join Time                      0.003376         0.002133            1.582           0.114
    L3.Buffer Ratio                  -0.001206         0.002122           -0.568           0.570
    L3.Connection Type                0.002345         0.001863            1.258           0.208
    L3.Device                         0.013646         0.005290            2.580           0.010
    L3.Device Type                   -0.012146         0.004319           -2.812           0.005
    L3.Browser                       -0.003114         0.006327           -0.492           0.623
    L3.OS                             0.064942         0.019745            3.289           0.001
    L3.OS Version                     0.000131         0.002064            0.063           0.949
    L3.Device ID                      0.000007         0.000015            0.490           0.624
    L3.Happiness Score                0.002429         0.001762            1.379           0.168
    L3.Playback Stalls                0.009707         0.030774            0.315           0.752
    L3.Startup Error (Count)          0.167040         0.148301            1.126           0.260
    L3.Latency                       -0.000000         0.000000           -0.367           0.713
    L3.Crash Status                  -0.091647         0.090929           -1.008           0.314
    L3.End of Playback Status         0.061944         0.039215            1.580           0.114
    L3.User_ID_N                      0.000011         0.000047            0.246           0.806
    L3.Title_N                        0.000008         0.000012            0.666           0.505
    L3.Device_Vendor_N               -0.000163         0.002612           -0.062           0.950
    L3.Device_Model_N                -0.000310         0.000307           -1.008           0.313
    L3.Content_TV_Show_N              0.000021         0.000012            1.721           0.085
    L3.Country_N                      0.008598         0.003978            2.161           0.031
    L3.City_N                         0.000004         0.000075            0.052           0.959
    L3.Region_N                      -0.272437         0.074173           -3.673           0.000
    L4.Playtime                       0.000004         0.000004            1.043           0.297
    L4.Interruptions                 -0.000304         0.000467           -0.651           0.515
    L4.Join Time                     -0.000387         0.002122           -0.182           0.855
    L4.Buffer Ratio                   0.004047         0.002122            1.907           0.056
    L4.Connection Type                0.003944         0.001871            2.108           0.035
    L4.Device                        -0.006351         0.005302           -1.198           0.231
    L4.Device Type                   -0.000692         0.004334           -0.160           0.873
    L4.Browser                       -0.005797         0.006349           -0.913           0.361
    L4.OS                             0.096335         0.019799            4.866           0.000
    L4.OS Version                    -0.001591         0.002069           -0.769           0.442
    L4.Device ID                      0.000011         0.000015            0.785           0.433
    L4.Happiness Score               -0.001296         0.001763           -0.735           0.462
    L4.Playback Stalls               -0.005983         0.030762           -0.195           0.846
    L4.Startup Error (Count)          0.018185         0.148293            0.123           0.902
    L4.Latency                       -0.000000         0.000000           -0.151           0.880
    L4.Crash Status                   0.064530         0.090929            0.710           0.478
    L4.End of Playback Status         0.034515         0.039223            0.880           0.379
    L4.User_ID_N                      0.000045         0.000047            0.956           0.339
    L4.Title_N                       -0.000009         0.000012           -0.733           0.464
    L4.Device_Vendor_N                0.003817         0.002621            1.456           0.145
    L4.Device_Model_N                 0.000062         0.000308            0.201           0.841
    L4.Content_TV_Show_N              0.000019         0.000012            1.582           0.114
    L4.Country_N                      0.004446         0.003988            1.115           0.265
    L4.City_N                         0.000114         0.000075            1.520           0.128
    L4.Region_N                      -0.155404         0.074383           -2.089           0.037
    L5.Playtime                      -0.000005         0.000004           -1.287           0.198
    L5.Interruptions                 -0.000065         0.000467           -0.140           0.889
    L5.Join Time                      0.004054         0.002123            1.910           0.056
    L5.Buffer Ratio                   0.002507         0.002119            1.183           0.237
    L5.Connection Type                0.000018         0.001872            0.010           0.992
    L5.Device                        -0.005310         0.005306           -1.001           0.317
    L5.Device Type                    0.001613         0.004339            0.372           0.710
    L5.Browser                        0.006669         0.006359            1.049           0.294
    L5.OS                             0.115166         0.019819            5.811           0.000
    L5.OS Version                    -0.008257         0.002072           -3.986           0.000
    L5.Device ID                      0.000011         0.000015            0.779           0.436
    L5.Happiness Score                0.001640         0.001764            0.930           0.352
    L5.Playback Stalls                0.027773         0.029954            0.927           0.354
    L5.Startup Error (Count)          0.011563         0.148279            0.078           0.938
    L5.Latency                       -0.000000         0.000000           -0.052           0.959
    L5.Crash Status                   0.048729         0.090941            0.536           0.592
    L5.End of Playback Status         0.025346         0.039239            0.646           0.518
    L5.User_ID_N                     -0.000073         0.000047           -1.559           0.119
    L5.Title_N                       -0.000010         0.000012           -0.801           0.423
    L5.Device_Vendor_N               -0.002698         0.002623           -1.029           0.304
    L5.Device_Model_N                -0.000360         0.000308           -1.168           0.243
    L5.Content_TV_Show_N              0.000005         0.000012            0.386           0.700
    L5.Country_N                      0.004992         0.003996            1.250           0.211
    L5.City_N                         0.000106         0.000075            1.417           0.157
    L5.Region_N                      -0.114965         0.074484           -1.543           0.123
    L6.Playtime                      -0.000001         0.000004           -0.217           0.828
    L6.Interruptions                 -0.000133         0.000467           -0.285           0.775
    L6.Join Time                      0.000483         0.002124            0.228           0.820
    L6.Buffer Ratio                   0.000854         0.002119            0.403           0.687
    L6.Connection Type                0.002160         0.001871            1.155           0.248
    L6.Device                         0.000426         0.005303            0.080           0.936
    L6.Device Type                   -0.003765         0.004335           -0.868           0.385
    L6.Browser                       -0.018210         0.006348           -2.868           0.004
    L6.OS                             0.037334         0.019799            1.886           0.059
    L6.OS Version                     0.005312         0.002070            2.566           0.010
    L6.Device ID                     -0.000004         0.000015           -0.297           0.766
    L6.Happiness Score               -0.001710         0.001762           -0.970           0.332
    L6.Playback Stalls                0.057650         0.030696            1.878           0.060
    L6.Startup Error (Count)          0.049280         0.148259            0.332           0.740
    L6.Latency                       -0.000000         0.000000           -0.748           0.455
    L6.Crash Status                   0.026040         0.090935            0.286           0.775
    L6.End of Playback Status        -0.005146         0.039244           -0.131           0.896
    L6.User_ID_N                     -0.000009         0.000047           -0.195           0.845
    L6.Title_N                        0.000012         0.000012            0.974           0.330
    L6.Device_Vendor_N                0.005266         0.002621            2.010           0.044
    L6.Device_Model_N                 0.000061         0.000308            0.197           0.843
    L6.Content_TV_Show_N             -0.000013         0.000012           -1.023           0.307
    L6.Country_N                      0.013613         0.003987            3.414           0.001
    L6.City_N                         0.000047         0.000075            0.624           0.533
    L6.Region_N                      -0.400640         0.074373           -5.387           0.000
    L7.Playtime                      -0.000001         0.000004           -0.328           0.743
    L7.Interruptions                 -0.000123         0.000467           -0.263           0.793
    L7.Join Time                     -0.000618         0.002123           -0.291           0.771
    L7.Buffer Ratio                  -0.002182         0.002119           -1.030           0.303
    L7.Connection Type                0.000956         0.001863            0.513           0.608
    L7.Device                         0.005228         0.005288            0.989           0.323
    L7.Device Type                   -0.008530         0.004319           -1.975           0.048
    L7.Browser                        0.011270         0.006328            1.781           0.075
    L7.OS                             0.052356         0.019744            2.652           0.008
    L7.OS Version                    -0.004712         0.002065           -2.282           0.022
    L7.Device ID                      0.000008         0.000015            0.570           0.569
    L7.Happiness Score               -0.002423         0.001760           -1.377           0.169
    L7.Playback Stalls               -0.002841         0.030855           -0.092           0.927
    L7.Startup Error (Count)          0.096080         0.148246            0.648           0.517
    L7.Latency                        0.000000         0.000000            1.413           0.158
    L7.Crash Status                  -0.142822         0.090933           -1.571           0.116
    L7.End of Playback Status        -0.023026         0.039239           -0.587           0.557
    L7.User_ID_N                     -0.000055         0.000047           -1.187           0.235
    L7.Title_N                        0.000002         0.000012            0.142           0.887
    L7.Device_Vendor_N                0.000006         0.002612            0.002           0.998
    L7.Device_Model_N                -0.000439         0.000307           -1.429           0.153
    L7.Content_TV_Show_N             -0.000010         0.000012           -0.801           0.423
    L7.Country_N                      0.003434         0.003977            0.863           0.388
    L7.City_N                         0.000017         0.000075            0.225           0.822
    L7.Region_N                      -0.147220         0.074169           -1.985           0.047
    L8.Playtime                       0.000003         0.000004            0.772           0.440
    L8.Interruptions                 -0.000158         0.000467           -0.338           0.736
    L8.Join Time                      0.003504         0.002123            1.650           0.099
    L8.Buffer Ratio                  -0.001651         0.002119           -0.779           0.436
    L8.Connection Type               -0.002294         0.001849           -1.241           0.215
    L8.Device                        -0.010718         0.005261           -2.037           0.042
    L8.Device Type                    0.009502         0.004286            2.217           0.027
    L8.Browser                       -0.014766         0.006294           -2.346           0.019
    L8.OS                             0.091471         0.019671            4.650           0.000
    L8.OS Version                    -0.003338         0.002058           -1.621           0.105
    L8.Device ID                      0.000028         0.000014            1.909           0.056
    L8.Happiness Score               -0.000065         0.001755           -0.037           0.970
    L8.Playback Stalls                0.016170         0.030856            0.524           0.600
    L8.Startup Error (Count)         -0.012271         0.148141           -0.083           0.934
    L8.Latency                       -0.000001         0.000000           -3.007           0.003
    L8.Crash Status                  -0.003715         0.090950           -0.041           0.967
    L8.End of Playback Status        -0.041386         0.039229           -1.055           0.291
    L8.User_ID_N                     -0.000015         0.000046           -0.335           0.738
    L8.Title_N                        0.000008         0.000012            0.674           0.500
    L8.Device_Vendor_N               -0.004735         0.002594           -1.826           0.068
    L8.Device_Model_N                -0.000295         0.000305           -0.965           0.334
    L8.Content_TV_Show_N             -0.000007         0.000012           -0.537           0.591
    L8.Country_N                      0.009205         0.003954            2.328           0.020
    L8.City_N                         0.000016         0.000074            0.217           0.828
    L8.Region_N                      -0.159340         0.073863           -2.157           0.031
    L9.Playtime                      -0.000001         0.000004           -0.280           0.779
    L9.Interruptions                 -0.000128         0.000467           -0.274           0.784
    L9.Join Time                      0.000961         0.002122            0.453           0.651
    L9.Buffer Ratio                   0.000010         0.002118            0.005           0.996
    L9.Connection Type               -0.001445         0.001826           -0.792           0.429
    L9.Device                        -0.001151         0.005209           -0.221           0.825
    L9.Device Type                    0.000831         0.004226            0.197           0.844
    L9.Browser                       -0.000273         0.006204           -0.044           0.965
    L9.OS                             0.065646         0.019441            3.377           0.001
    L9.OS Version                    -0.003341         0.002044           -1.635           0.102
    L9.Device ID                      0.000016         0.000014            1.130           0.258
    L9.Happiness Score               -0.001377         0.001731           -0.795           0.427
    L9.Playback Stalls               -0.055817         0.030749           -1.815           0.069
    L9.Startup Error (Count)         -0.293844         0.148007           -1.985           0.047
    L9.Latency                       -0.000000         0.000000           -0.722           0.470
    L9.Crash Status                   0.130627         0.090875            1.437           0.151
    L9.End of Playback Status        -0.040129         0.039215           -1.023           0.306
    L9.User_ID_N                     -0.000022         0.000045           -0.479           0.632
    L9.Title_N                        0.000008         0.000012            0.611           0.541
    L9.Device_Vendor_N                0.000473         0.002560            0.185           0.853
    L9.Device_Model_N                -0.000602         0.000302           -1.995           0.046
    L9.Content_TV_Show_N             -0.000012         0.000012           -1.011           0.312
    L9.Country_N                      0.013120         0.003927            3.341           0.001
    L9.City_N                        -0.000040         0.000073           -0.546           0.585
    L9.Region_N                      -0.141297         0.073390           -1.925           0.054
    ============================================================================================
    
    Results for equation OS Version
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             1.924229         2.309075            0.833           0.405
    L1.Playtime                       0.000009         0.000028            0.318           0.750
    L1.Interruptions                 -0.002736         0.003613           -0.757           0.449
    L1.Join Time                     -0.008992         0.016496           -0.545           0.586
    L1.Buffer Ratio                   0.018216         0.016365            1.113           0.266
    L1.Connection Type                0.014388         0.014118            1.019           0.308
    L1.Device                         0.217636         0.040312            5.399           0.000
    L1.Device Type                   -0.039690         0.032703           -1.214           0.225
    L1.Browser                       -0.014367         0.047974           -0.299           0.765
    L1.OS                            -0.461125         0.150527           -3.063           0.002
    L1.OS Version                     0.151297         0.015813            9.568           0.000
    L1.Device ID                     -0.000139         0.000111           -1.258           0.208
    L1.Happiness Score                0.022099         0.013407            1.648           0.099
    L1.Playback Stalls               -0.247160         0.237983           -1.039           0.299
    L1.Startup Error (Count)          2.654787         1.146314            2.316           0.021
    L1.Latency                        0.000001         0.000002            0.284           0.777
    L1.Crash Status                  -2.034069         0.701730           -2.899           0.004
    L1.End of Playback Status         0.590822         0.302319            1.954           0.051
    L1.User_ID_N                      0.000867         0.000352            2.465           0.014
    L1.Title_N                       -0.000065         0.000095           -0.683           0.494
    L1.Device_Vendor_N                0.023343         0.019802            1.179           0.238
    L1.Device_Model_N                 0.005080         0.002335            2.176           0.030
    L1.Content_TV_Show_N             -0.000526         0.000095           -5.569           0.000
    L1.Country_N                     -0.100725         0.030374           -3.316           0.001
    L1.City_N                        -0.002442         0.000564           -4.332           0.000
    L1.Region_N                       1.896303         0.567463            3.342           0.001
    L2.Playtime                      -0.000024         0.000028           -0.857           0.392
    L2.Interruptions                 -0.001275         0.003612           -0.353           0.724
    L2.Join Time                     -0.019089         0.016501           -1.157           0.247
    L2.Buffer Ratio                   0.012594         0.016404            0.768           0.443
    L2.Connection Type                0.005156         0.014296            0.361           0.718
    L2.Device                         0.032060         0.040704            0.788           0.431
    L2.Device Type                   -0.061161         0.033147           -1.845           0.065
    L2.Browser                        0.037511         0.048662            0.771           0.441
    L2.OS                            -0.013247         0.152169           -0.087           0.931
    L2.OS Version                     0.090358         0.015923            5.675           0.000
    L2.Device ID                      0.000003         0.000112            0.026           0.979
    L2.Happiness Score                0.030500         0.013584            2.245           0.025
    L2.Playback Stalls                0.209014         0.237999            0.878           0.380
    L2.Startup Error (Count)          2.078706         1.146972            1.812           0.070
    L2.Latency                        0.000001         0.000002            0.311           0.756
    L2.Crash Status                  -1.135244         0.703475           -1.614           0.107
    L2.End of Playback Status         0.623065         0.302983            2.056           0.040
    L2.User_ID_N                      0.000297         0.000357            0.833           0.405
    L2.Title_N                        0.000063         0.000096            0.660           0.509
    L2.Device_Vendor_N               -0.033935         0.020059           -1.692           0.091
    L2.Device_Model_N                 0.002361         0.002363            0.999           0.318
    L2.Content_TV_Show_N             -0.000071         0.000095           -0.746           0.456
    L2.Country_N                      0.005762         0.030592            0.188           0.851
    L2.City_N                        -0.001350         0.000572           -2.359           0.018
    L2.Region_N                      -0.210175         0.571335           -0.368           0.713
    L3.Playtime                      -0.000022         0.000028           -0.779           0.436
    L3.Interruptions                 -0.000995         0.003612           -0.276           0.783
    L3.Join Time                      0.024738         0.016500            1.499           0.134
    L3.Buffer Ratio                  -0.007054         0.016411           -0.430           0.667
    L3.Connection Type                0.016749         0.014413            1.162           0.245
    L3.Device                         0.116781         0.040912            2.854           0.004
    L3.Device Type                   -0.076451         0.033401           -2.289           0.022
    L3.Browser                       -0.006044         0.048935           -0.124           0.902
    L3.OS                            -0.164849         0.152713           -1.079           0.280
    L3.OS Version                     0.079768         0.015965            4.997           0.000
    L3.Device ID                      0.000046         0.000113            0.409           0.683
    L3.Happiness Score                0.026664         0.013625            1.957           0.050
    L3.Playback Stalls                0.004962         0.238012            0.021           0.983
    L3.Startup Error (Count)          0.938408         1.146992            0.818           0.413
    L3.Latency                       -0.000001         0.000002           -0.717           0.473
    L3.Crash Status                  -0.295045         0.703269           -0.420           0.675
    L3.End of Playback Status         0.282083         0.303297            0.930           0.352
    L3.User_ID_N                     -0.000058         0.000360           -0.162           0.871
    L3.Title_N                        0.000070         0.000096            0.735           0.462
    L3.Device_Vendor_N                0.003657         0.020200            0.181           0.856
    L3.Device_Model_N                -0.001253         0.002375           -0.528           0.598
    L3.Content_TV_Show_N              0.000167         0.000095            1.759           0.079
    L3.Country_N                      0.047272         0.030767            1.536           0.124
    L3.City_N                         0.000108         0.000577            0.188           0.851
    L3.Region_N                      -2.024850         0.573671           -3.530           0.000
    L4.Playtime                       0.000014         0.000028            0.491           0.624
    L4.Interruptions                 -0.001617         0.003612           -0.448           0.654
    L4.Join Time                     -0.002358         0.016412           -0.144           0.886
    L4.Buffer Ratio                   0.036459         0.016410            2.222           0.026
    L4.Connection Type                0.027536         0.014470            1.903           0.057
    L4.Device                        -0.034004         0.041007           -0.829           0.407
    L4.Device Type                    0.006840         0.033520            0.204           0.838
    L4.Browser                       -0.016106         0.049102           -0.328           0.743
    L4.OS                             0.272430         0.153129            1.779           0.075
    L4.OS Version                     0.038400         0.016006            2.399           0.016
    L4.Device ID                      0.000063         0.000113            0.559           0.576
    L4.Happiness Score               -0.009534         0.013639           -0.699           0.485
    L4.Playback Stalls               -0.111806         0.237917           -0.470           0.638
    L4.Startup Error (Count)          0.149116         1.146935            0.130           0.897
    L4.Latency                       -0.000000         0.000002           -0.105           0.916
    L4.Crash Status                   0.410642         0.703266            0.584           0.559
    L4.End of Playback Status         0.314303         0.303362            1.036           0.300
    L4.User_ID_N                      0.000257         0.000361            0.711           0.477
    L4.Title_N                       -0.000055         0.000096           -0.576           0.564
    L4.Device_Vendor_N                0.027285         0.020270            1.346           0.178
    L4.Device_Model_N                -0.000727         0.002381           -0.305           0.760
    L4.Content_TV_Show_N              0.000105         0.000095            1.102           0.270
    L4.Country_N                      0.014684         0.030845            0.476           0.634
    L4.City_N                         0.001339         0.000578            2.315           0.021
    L4.Region_N                      -1.161735         0.575296           -2.019           0.043
    L5.Playtime                      -0.000014         0.000028           -0.498           0.618
    L5.Interruptions                 -0.000700         0.003612           -0.194           0.846
    L5.Join Time                      0.022995         0.016419            1.401           0.161
    L5.Buffer Ratio                   0.014297         0.016386            0.872           0.383
    L5.Connection Type               -0.006346         0.014482           -0.438           0.661
    L5.Device                        -0.071453         0.041036           -1.741           0.082
    L5.Device Type                    0.038128         0.033561            1.136           0.256
    L5.Browser                        0.044175         0.049179            0.898           0.369
    L5.OS                             0.589272         0.153281            3.844           0.000
    L5.OS Version                    -0.021682         0.016022           -1.353           0.176
    L5.Device ID                      0.000095         0.000113            0.838           0.402
    L5.Happiness Score                0.015773         0.013640            1.156           0.248
    L5.Playback Stalls                0.081827         0.231673            0.353           0.724
    L5.Startup Error (Count)          0.588656         1.146828            0.513           0.608
    L5.Latency                       -0.000002         0.000002           -0.900           0.368
    L5.Crash Status                   0.112867         0.703361            0.160           0.873
    L5.End of Playback Status         0.129969         0.303481            0.428           0.668
    L5.User_ID_N                     -0.000569         0.000361           -1.575           0.115
    L5.Title_N                       -0.000046         0.000096           -0.480           0.631
    L5.Device_Vendor_N               -0.022759         0.020284           -1.122           0.262
    L5.Device_Model_N                -0.002500         0.002382           -1.049           0.294
    L5.Content_TV_Show_N             -0.000041         0.000095           -0.429           0.668
    L5.Country_N                      0.037175         0.030902            1.203           0.229
    L5.City_N                         0.000770         0.000579            1.329           0.184
    L5.Region_N                      -1.024728         0.576077           -1.779           0.075
    L6.Playtime                       0.000003         0.000028            0.106           0.916
    L6.Interruptions                 -0.001106         0.003612           -0.306           0.760
    L6.Join Time                      0.002611         0.016425            0.159           0.874
    L6.Buffer Ratio                   0.004635         0.016387            0.283           0.777
    L6.Connection Type                0.018504         0.014470            1.279           0.201
    L6.Device                        -0.004452         0.041011           -0.109           0.914
    L6.Device Type                   -0.010067         0.033529           -0.300           0.764
    L6.Browser                       -0.130442         0.049100           -2.657           0.008
    L6.OS                            -0.022928         0.153130           -0.150           0.881
    L6.OS Version                     0.079208         0.016007            4.948           0.000
    L6.Device ID                     -0.000041         0.000113           -0.359           0.719
    L6.Happiness Score               -0.015462         0.013631           -1.134           0.257
    L6.Playback Stalls                0.348522         0.237409            1.468           0.142
    L6.Startup Error (Count)          0.987136         1.146667            0.861           0.389
    L6.Latency                       -0.000002         0.000002           -0.857           0.392
    L6.Crash Status                  -0.121354         0.703311           -0.173           0.863
    L6.End of Playback Status         0.158856         0.303525            0.523           0.601
    L6.User_ID_N                     -0.000044         0.000361           -0.123           0.902
    L6.Title_N                        0.000050         0.000096            0.526           0.599
    L6.Device_Vendor_N                0.053961         0.020268            2.662           0.008
    L6.Device_Model_N                 0.001612         0.002380            0.677           0.498
    L6.Content_TV_Show_N             -0.000100         0.000095           -1.053           0.292
    L6.Country_N                      0.087068         0.030840            2.823           0.005
    L6.City_N                         0.000112         0.000579            0.193           0.847
    L6.Region_N                      -2.645445         0.575219           -4.599           0.000
    L7.Playtime                       0.000017         0.000028            0.594           0.553
    L7.Interruptions                 -0.001205         0.003612           -0.333           0.739
    L7.Join Time                      0.006052         0.016423            0.368           0.713
    L7.Buffer Ratio                  -0.019710         0.016389           -1.203           0.229
    L7.Connection Type                0.013351         0.014412            0.926           0.354
    L7.Device                         0.049570         0.040900            1.212           0.226
    L7.Device Type                   -0.086333         0.033402           -2.585           0.010
    L7.Browser                        0.104940         0.048943            2.144           0.032
    L7.OS                             0.017448         0.152702            0.114           0.909
    L7.OS Version                     0.007983         0.015967            0.500           0.617
    L7.Device ID                      0.000024         0.000113            0.216           0.829
    L7.Happiness Score               -0.010567         0.013612           -0.776           0.438
    L7.Playback Stalls               -0.058667         0.238643           -0.246           0.806
    L7.Startup Error (Count)          1.396176         1.146570            1.218           0.223
    L7.Latency                        0.000001         0.000002            0.394           0.693
    L7.Crash Status                  -1.640093         0.703298           -2.332           0.020
    L7.End of Playback Status         0.005218         0.303484            0.017           0.986
    L7.User_ID_N                     -0.000469         0.000360           -1.304           0.192
    L7.Title_N                       -0.000010         0.000096           -0.106           0.915
    L7.Device_Vendor_N                0.016899         0.020203            0.836           0.403
    L7.Device_Model_N                -0.001586         0.002375           -0.668           0.504
    L7.Content_TV_Show_N             -0.000036         0.000095           -0.373           0.709
    L7.Country_N                      0.016131         0.030757            0.524           0.600
    L7.City_N                         0.000468         0.000577            0.812           0.417
    L7.Region_N                      -1.625267         0.573639           -2.833           0.005
    L8.Playtime                       0.000030         0.000028            1.061           0.289
    L8.Interruptions                 -0.001374         0.003612           -0.380           0.704
    L8.Join Time                      0.013331         0.016422            0.812           0.417
    L8.Buffer Ratio                  -0.005667         0.016387           -0.346           0.729
    L8.Connection Type               -0.019381         0.014298           -1.355           0.175
    L8.Device                        -0.098958         0.040692           -2.432           0.015
    L8.Device Type                    0.069782         0.033150            2.105           0.035
    L8.Browser                       -0.084721         0.048683           -1.740           0.082
    L8.OS                             0.446287         0.152138            2.933           0.003
    L8.OS Version                     0.003817         0.015920            0.240           0.811
    L8.Device ID                      0.000233         0.000112            2.080           0.038
    L8.Happiness Score                0.002787         0.013575            0.205           0.837
    L8.Playback Stalls                0.030295         0.238648            0.127           0.899
    L8.Startup Error (Count)          0.301259         1.145754            0.263           0.793
    L8.Latency                       -0.000005         0.000002           -2.639           0.008
    L8.Crash Status                  -0.149209         0.703431           -0.212           0.832
    L8.End of Playback Status        -0.142845         0.303409           -0.471           0.638
    L8.User_ID_N                     -0.000085         0.000357           -0.239           0.811
    L8.Title_N                        0.000090         0.000095            0.946           0.344
    L8.Device_Vendor_N               -0.033316         0.020060           -1.661           0.097
    L8.Device_Model_N                -0.002113         0.002362           -0.894           0.371
    L8.Content_TV_Show_N             -0.000021         0.000095           -0.222           0.824
    L8.Country_N                      0.074742         0.030582            2.444           0.015
    L8.City_N                         0.000107         0.000572            0.187           0.852
    L8.Region_N                      -1.238837         0.571274           -2.169           0.030
    L9.Playtime                      -0.000008         0.000028           -0.292           0.770
    L9.Interruptions                 -0.000720         0.003612           -0.199           0.842
    L9.Join Time                      0.016084         0.016412            0.980           0.327
    L9.Buffer Ratio                  -0.000205         0.016383           -0.012           0.990
    L9.Connection Type               -0.016548         0.014119           -1.172           0.241
    L9.Device                        -0.013598         0.040287           -0.338           0.736
    L9.Device Type                    0.017446         0.032685            0.534           0.594
    L9.Browser                       -0.000524         0.047982           -0.011           0.991
    L9.OS                             0.213729         0.150363            1.421           0.155
    L9.OS Version                     0.013817         0.015808            0.874           0.382
    L9.Device ID                      0.000120         0.000111            1.082           0.279
    L9.Happiness Score               -0.008942         0.013391           -0.668           0.504
    L9.Playback Stalls               -0.311641         0.237821           -1.310           0.190
    L9.Startup Error (Count)         -2.053431         1.144722           -1.794           0.073
    L9.Latency                       -0.000002         0.000002           -0.984           0.325
    L9.Crash Status                   0.618986         0.702845            0.881           0.378
    L9.End of Playback Status        -0.320285         0.303300           -1.056           0.291
    L9.User_ID_N                     -0.000249         0.000352           -0.708           0.479
    L9.Title_N                        0.000052         0.000095            0.544           0.586
    L9.Device_Vendor_N                0.008415         0.019802            0.425           0.671
    L9.Device_Model_N                -0.004321         0.002335           -1.851           0.064
    L9.Content_TV_Show_N             -0.000100         0.000094           -1.056           0.291
    L9.Country_N                      0.075304         0.030369            2.480           0.013
    L9.City_N                        -0.000298         0.000564           -0.528           0.598
    L9.Region_N                      -1.043813         0.567616           -1.839           0.066
    ============================================================================================
    
    Results for equation Device ID
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                           148.234966        80.388557            1.844           0.065
    L1.Playtime                      -0.002285         0.000980           -2.332           0.020
    L1.Interruptions                  0.043351         0.125783            0.345           0.730
    L1.Join Time                      0.331182         0.574297            0.577           0.564
    L1.Buffer Ratio                   0.143537         0.569749            0.252           0.801
    L1.Connection Type                4.570857         0.491493            9.300           0.000
    L1.Device                        -3.204567         1.403441           -2.283           0.022
    L1.Device Type                    2.790321         1.138531            2.451           0.014
    L1.Browser                        1.618177         1.670170            0.969           0.333
    L1.OS                            17.682691         5.240475            3.374           0.001
    L1.OS Version                    -1.737784         0.550522           -3.157           0.002
    L1.Device ID                      0.140757         0.003850           36.560           0.000
    L1.Happiness Score               -1.717860         0.466752           -3.680           0.000
    L1.Playback Stalls               -3.389423         8.285177           -0.409           0.682
    L1.Startup Error (Count)          8.842803        39.907979            0.222           0.825
    L1.Latency                       -0.000050         0.000072           -0.695           0.487
    L1.Crash Status                 -13.945060        24.430159           -0.571           0.568
    L1.End of Playback Status        15.821864        10.524981            1.503           0.133
    L1.User_ID_N                      0.005730         0.012242            0.468           0.640
    L1.Title_N                        0.027525         0.003320            8.290           0.000
    L1.Device_Vendor_N                0.691843         0.689381            1.004           0.316
    L1.Device_Model_N                -0.066149         0.081282           -0.814           0.416
    L1.Content_TV_Show_N              0.000370         0.003291            0.112           0.911
    L1.Country_N                     -2.911464         1.057432           -2.753           0.006
    L1.City_N                         0.076697         0.019626            3.908           0.000
    L1.Region_N                      30.810793        19.755747            1.560           0.119
    L2.Playtime                      -0.001512         0.000980           -1.542           0.123
    L2.Interruptions                  0.156518         0.125756            1.245           0.213
    L2.Join Time                     -0.989323         0.574462           -1.722           0.085
    L2.Buffer Ratio                   0.398915         0.571105            0.698           0.485
    L2.Connection Type                2.462385         0.497719            4.947           0.000
    L2.Device                        -2.018800         1.417089           -1.425           0.154
    L2.Device Type                    0.724815         1.153973            0.628           0.530
    L2.Browser                        0.280406         1.694113            0.166           0.869
    L2.OS                            17.286743         5.297641            3.263           0.001
    L2.OS Version                    -2.168496         0.554340           -3.912           0.000
    L2.Device ID                      0.127500         0.003896           32.728           0.000
    L2.Happiness Score               -0.032623         0.472902           -0.069           0.945
    L2.Playback Stalls              -13.302017         8.285745           -1.605           0.108
    L2.Startup Error (Count)        -26.746144        39.930903           -0.670           0.503
    L2.Latency                       -0.000052         0.000072           -0.716           0.474
    L2.Crash Status                  16.563206        24.490905            0.676           0.499
    L2.End of Playback Status         5.386415        10.548105            0.511           0.610
    L2.User_ID_N                     -0.009696         0.012431           -0.780           0.435
    L2.Title_N                        0.023223         0.003325            6.984           0.000
    L2.Device_Vendor_N               -1.201550         0.698337           -1.721           0.085
    L2.Device_Model_N                -0.211707         0.082252           -2.574           0.010
    L2.Content_TV_Show_N              0.003656         0.003304            1.106           0.269
    L2.Country_N                     -0.250508         1.065019           -0.235           0.814
    L2.City_N                        -0.018250         0.019926           -0.916           0.360
    L2.Region_N                      54.566129        19.890563            2.743           0.006
    L3.Playtime                       0.000918         0.000980            0.937           0.349
    L3.Interruptions                 -0.097100         0.125754           -0.772           0.440
    L3.Join Time                      0.392667         0.574442            0.684           0.494
    L3.Buffer Ratio                   0.497687         0.571329            0.871           0.384
    L3.Connection Type                1.857024         0.501765            3.701           0.000
    L3.Device                        -0.318040         1.424334           -0.223           0.823
    L3.Device Type                   -1.429377         1.162824           -1.229           0.219
    L3.Browser                        0.020474         1.703632            0.012           0.990
    L3.OS                             5.057236         5.316570            0.951           0.341
    L3.OS Version                    -0.191243         0.555794           -0.344           0.731
    L3.Device ID                      0.095491         0.003925           24.327           0.000
    L3.Happiness Score               -0.005101         0.474343           -0.011           0.991
    L3.Playback Stalls                7.524324         8.286203            0.908           0.364
    L3.Startup Error (Count)         16.226762        39.931583            0.406           0.684
    L3.Latency                        0.000026         0.000072            0.359           0.719
    L3.Crash Status                 -15.385782        24.483737           -0.628           0.530
    L3.End of Playback Status        -3.128578        10.559026           -0.296           0.767
    L3.User_ID_N                     -0.009477         0.012522           -0.757           0.449
    L3.Title_N                        0.010818         0.003327            3.252           0.001
    L3.Device_Vendor_N                0.652781         0.703252            0.928           0.353
    L3.Device_Model_N                -0.022972         0.082683           -0.278           0.781
    L3.Content_TV_Show_N              0.003255         0.003312            0.983           0.326
    L3.Country_N                     -1.740787         1.071128           -1.625           0.104
    L3.City_N                         0.004680         0.020077            0.233           0.816
    L3.Region_N                      32.384006        19.971896            1.621           0.105
    L4.Playtime                      -0.001414         0.000980           -1.442           0.149
    L4.Interruptions                  0.050081         0.125754            0.398           0.690
    L4.Join Time                     -0.043162         0.571384           -0.076           0.940
    L4.Buffer Ratio                  -1.164726         0.571290           -2.039           0.041
    L4.Connection Type               -0.924293         0.503754           -1.835           0.067
    L4.Device                        -2.680714         1.427631           -1.878           0.060
    L4.Device Type                   -0.355268         1.166960           -0.304           0.761
    L4.Browser                       -0.605949         1.709459           -0.354           0.723
    L4.OS                             3.844525         5.331048            0.721           0.471
    L4.OS Version                    -0.158206         0.557227           -0.284           0.776
    L4.Device ID                      0.062893         0.003938           15.970           0.000
    L4.Happiness Score               -0.491865         0.474817           -1.036           0.300
    L4.Playback Stalls               -0.073870         8.282877           -0.009           0.993
    L4.Startup Error (Count)        -47.303710        39.929611           -1.185           0.236
    L4.Latency                       -0.000015         0.000072           -0.214           0.830
    L4.Crash Status                  23.672427        24.483626            0.967           0.334
    L4.End of Playback Status       -10.807111        10.561294           -1.023           0.306
    L4.User_ID_N                     -0.031546         0.012562           -2.511           0.012
    L4.Title_N                        0.004443         0.003326            1.336           0.182
    L4.Device_Vendor_N               -0.217707         0.705677           -0.309           0.758
    L4.Device_Model_N                 0.076198         0.082890            0.919           0.358
    L4.Content_TV_Show_N             -0.000555         0.003315           -0.167           0.867
    L4.Country_N                      2.613580         1.073837            2.434           0.015
    L4.City_N                        -0.038638         0.020140           -1.918           0.055
    L4.Region_N                      -0.831216        20.028460           -0.042           0.967
    L5.Playtime                       0.000932         0.000981            0.950           0.342
    L5.Interruptions                  0.116017         0.125757            0.923           0.356
    L5.Join Time                     -0.402467         0.571603           -0.704           0.481
    L5.Buffer Ratio                  -0.120048         0.570461           -0.210           0.833
    L5.Connection Type               -0.833915         0.504177           -1.654           0.098
    L5.Device                        -0.340296         1.428633           -0.238           0.812
    L5.Device Type                   -0.348757         1.168388           -0.298           0.765
    L5.Browser                        0.253914         1.712117            0.148           0.882
    L5.OS                             1.584194         5.336368            0.297           0.767
    L5.OS Version                     0.176784         0.557791            0.317           0.751
    L5.Device ID                      0.049760         0.003941           12.628           0.000
    L5.Happiness Score               -0.258931         0.474879           -0.545           0.586
    L5.Playback Stalls                2.140274         8.065505            0.265           0.791
    L5.Startup Error (Count)         79.818196        39.925860            1.999           0.046
    L5.Latency                       -0.000015         0.000072           -0.210           0.834
    L5.Crash Status                 -71.494487        24.486942           -2.920           0.004
    L5.End of Playback Status         1.955724        10.565449            0.185           0.853
    L5.User_ID_N                      0.005543         0.012573            0.441           0.659
    L5.Title_N                        0.002092         0.003326            0.629           0.529
    L5.Device_Vendor_N                0.615250         0.706182            0.871           0.384
    L5.Device_Model_N                 0.070522         0.082942            0.850           0.395
    L5.Content_TV_Show_N              0.009541         0.003317            2.876           0.004
    L5.Country_N                     -0.113563         1.075835           -0.106           0.916
    L5.City_N                         0.005528         0.020167            0.274           0.784
    L5.Region_N                      -8.987210        20.055634           -0.448           0.654
    L6.Playtime                       0.000527         0.000981            0.538           0.591
    L6.Interruptions                 -0.023037         0.125757           -0.183           0.855
    L6.Join Time                      0.789636         0.571830            1.381           0.167
    L6.Buffer Ratio                  -0.280133         0.570485           -0.491           0.623
    L6.Connection Type               -0.100710         0.503758           -0.200           0.842
    L6.Device                         0.699283         1.427764            0.490           0.624
    L6.Device Type                   -0.001682         1.167290           -0.001           0.999
    L6.Browser                       -0.855538         1.709373           -0.500           0.617
    L6.OS                            -6.051126         5.331078           -1.135           0.256
    L6.OS Version                     0.612346         0.557271            1.099           0.272
    L6.Device ID                      0.049831         0.003938           12.655           0.000
    L6.Happiness Score                0.228583         0.474535            0.482           0.630
    L6.Playback Stalls               -3.842284         8.265193           -0.465           0.642
    L6.Startup Error (Count)        -54.275210        39.920275           -1.360           0.174
    L6.Latency                        0.000024         0.000072            0.333           0.739
    L6.Crash Status                  11.806551        24.485178            0.482           0.630
    L6.End of Playback Status        -6.161144        10.566981           -0.583           0.560
    L6.User_ID_N                     -0.011780         0.012562           -0.938           0.348
    L6.Title_N                        0.003939         0.003327            1.184           0.236
    L6.Device_Vendor_N               -1.726337         0.705600           -2.447           0.014
    L6.Device_Model_N                 0.036562         0.082875            0.441           0.659
    L6.Content_TV_Show_N              0.004960         0.003316            1.496           0.135
    L6.Country_N                      0.515691         1.073660            0.480           0.631
    L6.City_N                         0.012852         0.020141            0.638           0.523
    L6.Region_N                     -41.929431        20.025780           -2.094           0.036
    L7.Playtime                       0.001497         0.000980            1.527           0.127
    L7.Interruptions                  0.081215         0.125762            0.646           0.518
    L7.Join Time                     -0.219078         0.571762           -0.383           0.702
    L7.Buffer Ratio                   0.043679         0.570580            0.077           0.939
    L7.Connection Type               -1.133623         0.501754           -2.259           0.024
    L7.Device                        -0.517987         1.423907           -0.364           0.716
    L7.Device Type                    0.532065         1.162859            0.458           0.647
    L7.Browser                        3.302483         1.703897            1.938           0.053
    L7.OS                             1.517966         5.316190            0.286           0.775
    L7.OS Version                    -0.845639         0.555891           -1.521           0.128
    L7.Device ID                      0.047431         0.003925           12.085           0.000
    L7.Happiness Score                0.473851         0.473896            1.000           0.317
    L7.Playback Stalls                7.726934         8.308162            0.930           0.352
    L7.Startup Error (Count)         -8.333745        39.916888           -0.209           0.835
    L7.Latency                       -0.000014         0.000072           -0.191           0.849
    L7.Crash Status                  -4.178465        24.484733           -0.171           0.864
    L7.End of Playback Status        -1.409748        10.565528           -0.133           0.894
    L7.User_ID_N                     -0.002040         0.012521           -0.163           0.871
    L7.Title_N                        0.003513         0.003327            1.056           0.291
    L7.Device_Vendor_N               -1.136837         0.703344           -1.616           0.106
    L7.Device_Model_N                 0.025293         0.082668            0.306           0.760
    L7.Content_TV_Show_N              0.000391         0.003312            0.118           0.906
    L7.Country_N                      2.015817         1.070775            1.883           0.060
    L7.City_N                         0.005019         0.020077            0.250           0.803
    L7.Region_N                      -8.987584        19.970781           -0.450           0.653
    L8.Playtime                      -0.001174         0.000980           -1.197           0.231
    L8.Interruptions                 -0.099267         0.125759           -0.789           0.430
    L8.Join Time                     -0.519519         0.571704           -0.909           0.363
    L8.Buffer Ratio                  -0.400424         0.570491           -0.702           0.483
    L8.Connection Type               -0.301225         0.497766           -0.605           0.545
    L8.Device                         1.073112         1.416642            0.758           0.449
    L8.Device Type                   -0.532391         1.154104           -0.461           0.645
    L8.Browser                       -0.903488         1.694848           -0.533           0.594
    L8.OS                            -8.285004         5.296563           -1.564           0.118
    L8.OS Version                     0.916466         0.554252            1.654           0.098
    L8.Device ID                      0.042200         0.003895           10.834           0.000
    L8.Happiness Score                0.280804         0.472620            0.594           0.552
    L8.Playback Stalls               -1.362894         8.308338           -0.164           0.870
    L8.Startup Error (Count)         10.823141        39.888494            0.271           0.786
    L8.Latency                        0.000047         0.000072            0.653           0.514
    L8.Crash Status                  25.763926        24.489365            1.052           0.293
    L8.End of Playback Status         4.399645        10.562935            0.417           0.677
    L8.User_ID_N                      0.009116         0.012431            0.733           0.463
    L8.Title_N                        0.000248         0.003324            0.075           0.941
    L8.Device_Vendor_N               -0.171115         0.698387           -0.245           0.806
    L8.Device_Model_N                -0.073787         0.082240           -0.897           0.370
    L8.Content_TV_Show_N             -0.002703         0.003303           -0.818           0.413
    L8.Country_N                      2.071514         1.064694            1.946           0.052
    L8.City_N                         0.014701         0.019923            0.738           0.461
    L8.Region_N                     -44.654959        19.888436           -2.245           0.025
    L9.Playtime                       0.000472         0.000980            0.481           0.630
    L9.Interruptions                 -0.105675         0.125760           -0.840           0.401
    L9.Join Time                      0.321435         0.571384            0.563           0.574
    L9.Buffer Ratio                   0.314015         0.570373            0.551           0.582
    L9.Connection Type               -0.797469         0.491540           -1.622           0.105
    L9.Device                        -2.363554         1.402544           -1.685           0.092
    L9.Device Type                   -0.044732         1.137908           -0.039           0.969
    L9.Browser                        2.186092         1.670446            1.309           0.191
    L9.OS                            -4.744233         5.234748           -0.906           0.365
    L9.OS Version                     0.635780         0.550334            1.155           0.248
    L9.Device ID                      0.033105         0.003853            8.593           0.000
    L9.Happiness Score               -0.406475         0.466206           -0.872           0.383
    L9.Playback Stalls               -3.397668         8.279539           -0.410           0.682
    L9.Startup Error (Count)         36.985070        39.852545            0.928           0.353
    L9.Latency                       -0.000110         0.000072           -1.526           0.127
    L9.Crash Status                 -24.472281        24.468960           -1.000           0.317
    L9.End of Playback Status         8.409740        10.559136            0.796           0.426
    L9.User_ID_N                      0.007237         0.012248            0.591           0.555
    L9.Title_N                        0.005011         0.003319            1.510           0.131
    L9.Device_Vendor_N               -0.173144         0.689385           -0.251           0.802
    L9.Device_Model_N                 0.075582         0.081281            0.930           0.352
    L9.Content_TV_Show_N             -0.003021         0.003289           -0.918           0.358
    L9.Country_N                      0.895736         1.057267            0.847           0.397
    L9.City_N                         0.010090         0.019642            0.514           0.607
    L9.Region_N                     -14.363533        19.761077           -0.727           0.467
    ============================================================================================
    
    Results for equation Happiness Score
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             3.710637         0.691975            5.362           0.000
    L1.Playtime                       0.000018         0.000008            2.174           0.030
    L1.Interruptions                  0.000338         0.001083            0.312           0.755
    L1.Join Time                      0.015722         0.004943            3.180           0.001
    L1.Buffer Ratio                   0.009085         0.004904            1.852           0.064
    L1.Connection Type                0.004267         0.004231            1.009           0.313
    L1.Device                        -0.069161         0.012081           -5.725           0.000
    L1.Device Type                   -0.027427         0.009800           -2.799           0.005
    L1.Browser                        0.005746         0.014377            0.400           0.689
    L1.OS                             0.064927         0.045109            1.439           0.150
    L1.OS Version                     0.003545         0.004739            0.748           0.454
    L1.Device ID                     -0.000318         0.000033           -9.587           0.000
    L1.Happiness Score                0.168566         0.004018           41.955           0.000
    L1.Playback Stalls               -0.066712         0.071318           -0.935           0.350
    L1.Startup Error (Count)          0.689439         0.343523            2.007           0.045
    L1.Latency                       -0.000002         0.000001           -3.907           0.000
    L1.Crash Status                   0.006779         0.210292            0.032           0.974
    L1.End of Playback Status         0.205983         0.090598            2.274           0.023
    L1.User_ID_N                      0.000742         0.000105            7.040           0.000
    L1.Title_N                       -0.000094         0.000029           -3.287           0.001
    L1.Device_Vendor_N               -0.011522         0.005934           -1.942           0.052
    L1.Device_Model_N                 0.002078         0.000700            2.970           0.003
    L1.Content_TV_Show_N              0.000152         0.000028            5.370           0.000
    L1.Country_N                      0.037010         0.009102            4.066           0.000
    L1.City_N                        -0.001471         0.000169           -8.710           0.000
    L1.Region_N                       1.000979         0.170055            5.886           0.000
    L2.Playtime                      -0.000003         0.000008           -0.403           0.687
    L2.Interruptions                  0.000650         0.001082            0.600           0.548
    L2.Join Time                      0.010752         0.004945            2.174           0.030
    L2.Buffer Ratio                  -0.002923         0.004916           -0.595           0.552
    L2.Connection Type                0.020348         0.004284            4.749           0.000
    L2.Device                        -0.030216         0.012198           -2.477           0.013
    L2.Device Type                    0.027422         0.009933            2.761           0.006
    L2.Browser                       -0.011817         0.014583           -0.810           0.418
    L2.OS                             0.073118         0.045601            1.603           0.109
    L2.OS Version                    -0.000098         0.004772           -0.021           0.984
    L2.Device ID                     -0.000137         0.000034           -4.098           0.000
    L2.Happiness Score                0.083814         0.004071           20.590           0.000
    L2.Playback Stalls                0.066908         0.071323            0.938           0.348
    L2.Startup Error (Count)          0.385141         0.343721            1.121           0.262
    L2.Latency                        0.000002         0.000001            3.721           0.000
    L2.Crash Status                  -0.329199         0.210815           -1.562           0.118
    L2.End of Playback Status        -0.033986         0.090797           -0.374           0.708
    L2.User_ID_N                      0.000347         0.000107            3.245           0.001
    L2.Title_N                       -0.000032         0.000029           -1.119           0.263
    L2.Device_Vendor_N               -0.006977         0.006011           -1.161           0.246
    L2.Device_Model_N                -0.000309         0.000708           -0.436           0.663
    L2.Content_TV_Show_N              0.000001         0.000028            0.051           0.959
    L2.Country_N                     -0.003019         0.009168           -0.329           0.742
    L2.City_N                        -0.000271         0.000172           -1.580           0.114
    L2.Region_N                       0.498270         0.171216            2.910           0.004
    L3.Playtime                       0.000009         0.000008            1.031           0.302
    L3.Interruptions                  0.000245         0.001082            0.226           0.821
    L3.Join Time                     -0.002096         0.004945           -0.424           0.672
    L3.Buffer Ratio                  -0.001510         0.004918           -0.307           0.759
    L3.Connection Type                0.013049         0.004319            3.021           0.003
    L3.Device                        -0.016129         0.012261           -1.316           0.188
    L3.Device Type                    0.038845         0.010009            3.881           0.000
    L3.Browser                       -0.023829         0.014665           -1.625           0.104
    L3.OS                             0.044968         0.045764            0.983           0.326
    L3.OS Version                    -0.000358         0.004784           -0.075           0.940
    L3.Device ID                     -0.000122         0.000034           -3.614           0.000
    L3.Happiness Score                0.051867         0.004083           12.703           0.000
    L3.Playback Stalls               -0.092631         0.071327           -1.299           0.194
    L3.Startup Error (Count)          0.554525         0.343726            1.613           0.107
    L3.Latency                        0.000002         0.000001            3.688           0.000
    L3.Crash Status                  -0.414420         0.210753           -1.966           0.049
    L3.End of Playback Status        -0.025357         0.090891           -0.279           0.780
    L3.User_ID_N                     -0.000113         0.000108           -1.044           0.296
    L3.Title_N                        0.000009         0.000029            0.309           0.757
    L3.Device_Vendor_N               -0.012916         0.006054           -2.134           0.033
    L3.Device_Model_N                -0.000281         0.000712           -0.395           0.693
    L3.Content_TV_Show_N             -0.000063         0.000029           -2.195           0.028
    L3.Country_N                     -0.003223         0.009220           -0.350           0.727
    L3.City_N                        -0.000295         0.000173           -1.705           0.088
    L3.Region_N                       0.284790         0.171916            1.657           0.098
    L4.Playtime                       0.000014         0.000008            1.662           0.097
    L4.Interruptions                  0.000201         0.001082            0.186           0.852
    L4.Join Time                     -0.001904         0.004918           -0.387           0.699
    L4.Buffer Ratio                   0.002178         0.004918            0.443           0.658
    L4.Connection Type                0.014108         0.004336            3.253           0.001
    L4.Device                        -0.015712         0.012289           -1.279           0.201
    L4.Device Type                   -0.017599         0.010045           -1.752           0.080
    L4.Browser                        0.013678         0.014715            0.930           0.353
    L4.OS                            -0.031336         0.045889           -0.683           0.495
    L4.OS Version                    -0.002125         0.004797           -0.443           0.658
    L4.Device ID                     -0.000048         0.000034           -1.424           0.154
    L4.Happiness Score                0.035713         0.004087            8.738           0.000
    L4.Playback Stalls               -0.080050         0.071298           -1.123           0.262
    L4.Startup Error (Count)          0.204777         0.343709            0.596           0.551
    L4.Latency                        0.000002         0.000001            3.482           0.000
    L4.Crash Status                  -0.483312         0.210752           -2.293           0.022
    L4.End of Playback Status         0.074994         0.090910            0.825           0.409
    L4.User_ID_N                      0.000030         0.000108            0.279           0.780
    L4.Title_N                       -0.000124         0.000029           -4.319           0.000
    L4.Device_Vendor_N               -0.013318         0.006074           -2.192           0.028
    L4.Device_Model_N                 0.000321         0.000714            0.450           0.653
    L4.Content_TV_Show_N              0.000033         0.000029            1.165           0.244
    L4.Country_N                     -0.028648         0.009243           -3.099           0.002
    L4.City_N                         0.000138         0.000173            0.795           0.427
    L4.Region_N                       0.412111         0.172403            2.390           0.017
    L5.Playtime                       0.000019         0.000008            2.219           0.027
    L5.Interruptions                  0.000097         0.001083            0.090           0.928
    L5.Join Time                     -0.006897         0.004920           -1.402           0.161
    L5.Buffer Ratio                  -0.001854         0.004910           -0.378           0.706
    L5.Connection Type                0.008249         0.004340            1.901           0.057
    L5.Device                        -0.000569         0.012298           -0.046           0.963
    L5.Device Type                   -0.009106         0.010057           -0.905           0.365
    L5.Browser                        0.033042         0.014738            2.242           0.025
    L5.OS                            -0.085898         0.045935           -1.870           0.061
    L5.OS Version                     0.002138         0.004801            0.445           0.656
    L5.Device ID                      0.000002         0.000034            0.068           0.946
    L5.Happiness Score                0.029432         0.004088            7.200           0.000
    L5.Playback Stalls               -0.070067         0.069427           -1.009           0.313
    L5.Startup Error (Count)          0.074322         0.343677            0.216           0.829
    L5.Latency                        0.000001         0.000001            1.466           0.143
    L5.Crash Status                  -0.139901         0.210781           -0.664           0.507
    L5.End of Playback Status         0.024300         0.090946            0.267           0.789
    L5.User_ID_N                      0.000094         0.000108            0.871           0.384
    L5.Title_N                       -0.000133         0.000029           -4.631           0.000
    L5.Device_Vendor_N               -0.001966         0.006079           -0.323           0.746
    L5.Device_Model_N                 0.000822         0.000714            1.152           0.250
    L5.Content_TV_Show_N             -0.000000         0.000029           -0.010           0.992
    L5.Country_N                     -0.014250         0.009261           -1.539           0.124
    L5.City_N                        -0.000096         0.000174           -0.553           0.580
    L5.Region_N                       0.232027         0.172637            1.344           0.179
    L6.Playtime                       0.000004         0.000008            0.424           0.671
    L6.Interruptions                  0.000535         0.001083            0.494           0.621
    L6.Join Time                     -0.006221         0.004922           -1.264           0.206
    L6.Buffer Ratio                   0.002406         0.004911            0.490           0.624
    L6.Connection Type                0.005154         0.004336            1.189           0.235
    L6.Device                        -0.024975         0.012290           -2.032           0.042
    L6.Device Type                    0.009733         0.010048            0.969           0.333
    L6.Browser                        0.020652         0.014714            1.404           0.160
    L6.OS                             0.089531         0.045889            1.951           0.051
    L6.OS Version                    -0.013351         0.004797           -2.783           0.005
    L6.Device ID                     -0.000035         0.000034           -1.018           0.309
    L6.Happiness Score                0.031927         0.004085            7.816           0.000
    L6.Playback Stalls                0.002318         0.071146            0.033           0.974
    L6.Startup Error (Count)          0.144277         0.343629            0.420           0.675
    L6.Latency                        0.000001         0.000001            1.446           0.148
    L6.Crash Status                  -0.227578         0.210766           -1.080           0.280
    L6.End of Playback Status         0.028854         0.090959            0.317           0.751
    L6.User_ID_N                     -0.000130         0.000108           -1.202           0.230
    L6.Title_N                       -0.000134         0.000029           -4.676           0.000
    L6.Device_Vendor_N               -0.006606         0.006074           -1.088           0.277
    L6.Device_Model_N                 0.000334         0.000713            0.469           0.639
    L6.Content_TV_Show_N              0.000050         0.000029            1.742           0.081
    L6.Country_N                     -0.014157         0.009242           -1.532           0.126
    L6.City_N                         0.000028         0.000173            0.161           0.872
    L6.Region_N                       0.552652         0.172380            3.206           0.001
    L7.Playtime                       0.000003         0.000008            0.352           0.725
    L7.Interruptions                  0.000286         0.001083            0.265           0.791
    L7.Join Time                      0.004630         0.004922            0.941           0.347
    L7.Buffer Ratio                   0.000538         0.004911            0.110           0.913
    L7.Connection Type               -0.001884         0.004319           -0.436           0.663
    L7.Device                         0.010969         0.012257            0.895           0.371
    L7.Device Type                    0.009968         0.010010            0.996           0.319
    L7.Browser                       -0.012034         0.014667           -0.820           0.412
    L7.OS                            -0.091582         0.045761           -2.001           0.045
    L7.OS Version                     0.013114         0.004785            2.741           0.006
    L7.Device ID                     -0.000027         0.000034           -0.798           0.425
    L7.Happiness Score                0.045381         0.004079           11.125           0.000
    L7.Playback Stalls                0.000575         0.071516            0.008           0.994
    L7.Startup Error (Count)          0.064347         0.343600            0.187           0.851
    L7.Latency                       -0.000001         0.000001           -1.177           0.239
    L7.Crash Status                   0.098018         0.210762            0.465           0.642
    L7.End of Playback Status         0.030807         0.090947            0.339           0.735
    L7.User_ID_N                      0.000129         0.000108            1.196           0.232
    L7.Title_N                       -0.000010         0.000029           -0.342           0.732
    L7.Device_Vendor_N                0.002438         0.006054            0.403           0.687
    L7.Device_Model_N                 0.000060         0.000712            0.085           0.932
    L7.Content_TV_Show_N              0.000005         0.000029            0.160           0.873
    L7.Country_N                     -0.004710         0.009217           -0.511           0.609
    L7.City_N                        -0.000109         0.000173           -0.630           0.529
    L7.Region_N                       0.026079         0.171906            0.152           0.879
    L8.Playtime                      -0.000006         0.000008           -0.669           0.503
    L8.Interruptions                 -0.001552         0.001083           -1.433           0.152
    L8.Join Time                     -0.004244         0.004921           -0.862           0.388
    L8.Buffer Ratio                   0.005508         0.004911            1.122           0.262
    L8.Connection Type                0.006870         0.004285            1.603           0.109
    L8.Device                         0.011923         0.012194            0.978           0.328
    L8.Device Type                   -0.001145         0.009934           -0.115           0.908
    L8.Browser                        0.006403         0.014589            0.439           0.661
    L8.OS                            -0.012950         0.045592           -0.284           0.776
    L8.OS Version                    -0.000077         0.004771           -0.016           0.987
    L8.Device ID                      0.000023         0.000034            0.699           0.484
    L8.Happiness Score                0.031138         0.004068            7.654           0.000
    L8.Playback Stalls               -0.008490         0.071517           -0.119           0.906
    L8.Startup Error (Count)         -0.170273         0.343356           -0.496           0.620
    L8.Latency                        0.000000         0.000001            0.494           0.621
    L8.Crash Status                   0.070382         0.210802            0.334           0.738
    L8.End of Playback Status        -0.032524         0.090925           -0.358           0.721
    L8.User_ID_N                      0.000212         0.000107            1.978           0.048
    L8.Title_N                        0.000012         0.000029            0.436           0.663
    L8.Device_Vendor_N                0.002509         0.006012            0.417           0.676
    L8.Device_Model_N                -0.000961         0.000708           -1.357           0.175
    L8.Content_TV_Show_N              0.000007         0.000028            0.262           0.793
    L8.Country_N                     -0.023122         0.009165           -2.523           0.012
    L8.City_N                         0.000012         0.000171            0.073           0.942
    L8.Region_N                       0.214645         0.171197            1.254           0.210
    L9.Playtime                       0.000008         0.000008            0.955           0.340
    L9.Interruptions                  0.001164         0.001083            1.076           0.282
    L9.Join Time                     -0.002118         0.004918           -0.431           0.667
    L9.Buffer Ratio                   0.002417         0.004910            0.492           0.623
    L9.Connection Type               -0.003800         0.004231           -0.898           0.369
    L9.Device                         0.005706         0.012073            0.473           0.637
    L9.Device Type                   -0.001787         0.009795           -0.182           0.855
    L9.Browser                        0.003263         0.014379            0.227           0.820
    L9.OS                            -0.025017         0.045060           -0.555           0.579
    L9.OS Version                     0.001145         0.004737            0.242           0.809
    L9.Device ID                     -0.000032         0.000033           -0.968           0.333
    L9.Happiness Score                0.031511         0.004013            7.852           0.000
    L9.Playback Stalls                0.114562         0.071269            1.607           0.108
    L9.Startup Error (Count)         -0.126859         0.343046           -0.370           0.712
    L9.Latency                        0.000000         0.000001            0.209           0.834
    L9.Crash Status                  -0.101122         0.210626           -0.480           0.631
    L9.End of Playback Status        -0.125976         0.090892           -1.386           0.166
    L9.User_ID_N                     -0.000015         0.000105           -0.140           0.888
    L9.Title_N                        0.000017         0.000029            0.585           0.559
    L9.Device_Vendor_N               -0.000371         0.005934           -0.063           0.950
    L9.Device_Model_N                -0.000428         0.000700           -0.611           0.541
    L9.Content_TV_Show_N              0.000006         0.000028            0.197           0.844
    L9.Country_N                     -0.028405         0.009101           -3.121           0.002
    L9.City_N                        -0.000058         0.000169           -0.343           0.732
    L9.Region_N                       0.101167         0.170101            0.595           0.552
    ============================================================================================
    
    Results for equation Playback Stalls
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             0.032745         0.036749            0.891           0.373
    L1.Playtime                      -0.000000         0.000000           -0.439           0.661
    L1.Interruptions                 -0.000011         0.000058           -0.184           0.854
    L1.Join Time                      0.000990         0.000263            3.771           0.000
    L1.Buffer Ratio                   0.004491         0.000260           17.245           0.000
    L1.Connection Type                0.000073         0.000225            0.323           0.747
    L1.Device                         0.000067         0.000642            0.104           0.917
    L1.Device Type                    0.000036         0.000520            0.068           0.945
    L1.Browser                       -0.001276         0.000763           -1.671           0.095
    L1.OS                             0.000843         0.002396            0.352           0.725
    L1.OS Version                    -0.000003         0.000252           -0.012           0.990
    L1.Device ID                      0.000002         0.000002            0.899           0.368
    L1.Happiness Score               -0.000107         0.000213           -0.501           0.616
    L1.Playback Stalls                0.010594         0.003787            2.797           0.005
    L1.Startup Error (Count)          0.144467         0.018243            7.919           0.000
    L1.Latency                        0.000000         0.000000            0.267           0.789
    L1.Crash Status                  -0.110833         0.011168           -9.924           0.000
    L1.End of Playback Status         0.026821         0.004811            5.575           0.000
    L1.User_ID_N                      0.000001         0.000006            0.130           0.897
    L1.Title_N                        0.000002         0.000002            1.257           0.209
    L1.Device_Vendor_N                0.000031         0.000315            0.099           0.921
    L1.Device_Model_N                -0.000010         0.000037           -0.270           0.787
    L1.Content_TV_Show_N             -0.000000         0.000002           -0.236           0.813
    L1.Country_N                     -0.000361         0.000483           -0.747           0.455
    L1.City_N                        -0.000005         0.000009           -0.606           0.545
    L1.Region_N                       0.010488         0.009031            1.161           0.246
    L2.Playtime                       0.000000         0.000000            0.341           0.733
    L2.Interruptions                 -0.000004         0.000057           -0.064           0.949
    L2.Join Time                     -0.000111         0.000263           -0.422           0.673
    L2.Buffer Ratio                  -0.000091         0.000261           -0.347           0.728
    L2.Connection Type                0.000290         0.000228            1.273           0.203
    L2.Device                        -0.000121         0.000648           -0.186           0.852
    L2.Device Type                    0.000402         0.000528            0.761           0.446
    L2.Browser                       -0.000235         0.000774           -0.304           0.761
    L2.OS                             0.002433         0.002422            1.005           0.315
    L2.OS Version                    -0.000105         0.000253           -0.415           0.678
    L2.Device ID                     -0.000000         0.000002           -0.049           0.961
    L2.Happiness Score               -0.000301         0.000216           -1.395           0.163
    L2.Playback Stalls               -0.001100         0.003788           -0.290           0.772
    L2.Startup Error (Count)         -0.006409         0.018254           -0.351           0.726
    L2.Latency                       -0.000000         0.000000           -0.521           0.603
    L2.Crash Status                  -0.013682         0.011196           -1.222           0.222
    L2.End of Playback Status        -0.007920         0.004822           -1.642           0.100
    L2.User_ID_N                     -0.000000         0.000006           -0.017           0.987
    L2.Title_N                        0.000001         0.000002            0.772           0.440
    L2.Device_Vendor_N                0.000431         0.000319            1.349           0.177
    L2.Device_Model_N                -0.000026         0.000038           -0.688           0.492
    L2.Content_TV_Show_N             -0.000001         0.000002           -0.987           0.324
    L2.Country_N                     -0.000830         0.000487           -1.705           0.088
    L2.City_N                        -0.000006         0.000009           -0.681           0.496
    L2.Region_N                       0.006800         0.009093            0.748           0.455
    L3.Playtime                      -0.000000         0.000000           -0.583           0.560
    L3.Interruptions                  0.000004         0.000057            0.071           0.944
    L3.Join Time                      0.000166         0.000263            0.630           0.529
    L3.Buffer Ratio                  -0.000201         0.000261           -0.771           0.441
    L3.Connection Type               -0.000046         0.000229           -0.198           0.843
    L3.Device                        -0.000216         0.000651           -0.331           0.740
    L3.Device Type                   -0.000768         0.000532           -1.445           0.149
    L3.Browser                        0.000590         0.000779            0.758           0.449
    L3.OS                            -0.002869         0.002430           -1.181           0.238
    L3.OS Version                     0.000035         0.000254            0.137           0.891
    L3.Device ID                      0.000001         0.000002            0.741           0.458
    L3.Happiness Score               -0.000043         0.000217           -0.196           0.844
    L3.Playback Stalls               -0.000007         0.003788           -0.002           0.998
    L3.Startup Error (Count)          0.002071         0.018254            0.113           0.910
    L3.Latency                       -0.000000         0.000000           -0.188           0.851
    L3.Crash Status                   0.001751         0.011192            0.156           0.876
    L3.End of Playback Status         0.003381         0.004827            0.700           0.484
    L3.User_ID_N                      0.000004         0.000006            0.756           0.449
    L3.Title_N                        0.000002         0.000002            1.142           0.254
    L3.Device_Vendor_N               -0.000970         0.000321           -3.016           0.003
    L3.Device_Model_N                 0.000043         0.000038            1.128           0.259
    L3.Content_TV_Show_N             -0.000000         0.000002           -0.230           0.818
    L3.Country_N                      0.000485         0.000490            0.991           0.321
    L3.City_N                         0.000015         0.000009            1.651           0.099
    L3.Region_N                      -0.013085         0.009130           -1.433           0.152
    L4.Playtime                      -0.000000         0.000000           -0.620           0.535
    L4.Interruptions                  0.000002         0.000057            0.039           0.969
    L4.Join Time                      0.002295         0.000261            8.788           0.000
    L4.Buffer Ratio                  -0.000080         0.000261           -0.307           0.759
    L4.Connection Type               -0.000168         0.000230           -0.730           0.465
    L4.Device                        -0.000440         0.000653           -0.674           0.500
    L4.Device Type                   -0.000111         0.000533           -0.208           0.835
    L4.Browser                       -0.001051         0.000781           -1.345           0.179
    L4.OS                             0.001182         0.002437            0.485           0.628
    L4.OS Version                    -0.000032         0.000255           -0.127           0.899
    L4.Device ID                     -0.000001         0.000002           -0.791           0.429
    L4.Happiness Score               -0.000079         0.000217           -0.365           0.715
    L4.Playback Stalls                0.010444         0.003786            2.758           0.006
    L4.Startup Error (Count)         -0.012410         0.018253           -0.680           0.497
    L4.Latency                       -0.000000         0.000000           -0.265           0.791
    L4.Crash Status                   0.014342         0.011192            1.281           0.200
    L4.End of Playback Status        -0.006100         0.004828           -1.263           0.206
    L4.User_ID_N                     -0.000004         0.000006           -0.656           0.512
    L4.Title_N                       -0.000000         0.000002           -0.088           0.930
    L4.Device_Vendor_N               -0.000132         0.000323           -0.409           0.682
    L4.Device_Model_N                 0.000059         0.000038            1.556           0.120
    L4.Content_TV_Show_N             -0.000000         0.000002           -0.281           0.778
    L4.Country_N                      0.000201         0.000491            0.409           0.683
    L4.City_N                         0.000004         0.000009            0.444           0.657
    L4.Region_N                       0.000827         0.009156            0.090           0.928
    L5.Playtime                      -0.000000         0.000000           -0.336           0.737
    L5.Interruptions                  0.000007         0.000057            0.119           0.905
    L5.Join Time                      0.002508         0.000261            9.599           0.000
    L5.Buffer Ratio                  -0.000733         0.000261           -2.812           0.005
    L5.Connection Type               -0.000095         0.000230           -0.411           0.681
    L5.Device                         0.000257         0.000653            0.394           0.694
    L5.Device Type                    0.000404         0.000534            0.756           0.450
    L5.Browser                        0.000653         0.000783            0.834           0.404
    L5.OS                            -0.000852         0.002439           -0.349           0.727
    L5.OS Version                    -0.000141         0.000255           -0.555           0.579
    L5.Device ID                     -0.000002         0.000002           -0.931           0.352
    L5.Happiness Score                0.000085         0.000217            0.390           0.697
    L5.Playback Stalls                0.228566         0.003687           61.992           0.000
    L5.Startup Error (Count)          0.038180         0.018252            2.092           0.036
    L5.Latency                       -0.000000         0.000000           -1.053           0.292
    L5.Crash Status                  -0.012544         0.011194           -1.121           0.262
    L5.End of Playback Status         0.006037         0.004830            1.250           0.211
    L5.User_ID_N                     -0.000005         0.000006           -0.865           0.387
    L5.Title_N                        0.000001         0.000002            0.549           0.583
    L5.Device_Vendor_N               -0.000106         0.000323           -0.328           0.743
    L5.Device_Model_N                 0.000005         0.000038            0.131           0.896
    L5.Content_TV_Show_N              0.000001         0.000002            0.423           0.673
    L5.Country_N                     -0.000641         0.000492           -1.304           0.192
    L5.City_N                        -0.000004         0.000009           -0.425           0.671
    L5.Region_N                       0.015581         0.009168            1.699           0.089
    L6.Playtime                       0.000000         0.000000            0.004           0.997
    L6.Interruptions                  0.000006         0.000057            0.107           0.915
    L6.Join Time                     -0.000590         0.000261           -2.256           0.024
    L6.Buffer Ratio                  -0.001085         0.000261           -4.160           0.000
    L6.Connection Type                0.000073         0.000230            0.318           0.751
    L6.Device                         0.000528         0.000653            0.809           0.418
    L6.Device Type                    0.000457         0.000534            0.856           0.392
    L6.Browser                        0.001241         0.000781            1.588           0.112
    L6.OS                             0.001569         0.002437            0.644           0.520
    L6.OS Version                    -0.000277         0.000255           -1.086           0.277
    L6.Device ID                      0.000002         0.000002            1.333           0.182
    L6.Happiness Score                0.000042         0.000217            0.193           0.847
    L6.Playback Stalls               -0.008823         0.003778           -2.335           0.020
    L6.Startup Error (Count)         -0.046413         0.018249           -2.543           0.011
    L6.Latency                       -0.000000         0.000000           -0.433           0.665
    L6.Crash Status                   0.034020         0.011193            3.039           0.002
    L6.End of Playback Status        -0.007301         0.004831           -1.511           0.131
    L6.User_ID_N                      0.000008         0.000006            1.437           0.151
    L6.Title_N                       -0.000002         0.000002           -1.056           0.291
    L6.Device_Vendor_N                0.000032         0.000323            0.099           0.921
    L6.Device_Model_N                -0.000041         0.000038           -1.084           0.278
    L6.Content_TV_Show_N              0.000001         0.000002            0.824           0.410
    L6.Country_N                     -0.000045         0.000491           -0.091           0.927
    L6.City_N                        -0.000005         0.000009           -0.507           0.612
    L6.Region_N                      -0.003190         0.009155           -0.349           0.727
    L7.Playtime                      -0.000000         0.000000           -0.234           0.815
    L7.Interruptions                 -0.000001         0.000057           -0.020           0.984
    L7.Join Time                     -0.000147         0.000261           -0.561           0.575
    L7.Buffer Ratio                   0.000196         0.000261            0.752           0.452
    L7.Connection Type                0.000124         0.000229            0.542           0.588
    L7.Device                        -0.000486         0.000651           -0.747           0.455
    L7.Device Type                   -0.000144         0.000532           -0.270           0.787
    L7.Browser                       -0.000856         0.000779           -1.098           0.272
    L7.OS                             0.001558         0.002430            0.641           0.522
    L7.OS Version                    -0.000054         0.000254           -0.213           0.831
    L7.Device ID                     -0.000001         0.000002           -0.295           0.768
    L7.Happiness Score               -0.000437         0.000217           -2.019           0.044
    L7.Playback Stalls               -0.000786         0.003798           -0.207           0.836
    L7.Startup Error (Count)         -0.004200         0.018247           -0.230           0.818
    L7.Latency                        0.000000         0.000000            0.076           0.939
    L7.Crash Status                   0.005017         0.011193            0.448           0.654
    L7.End of Playback Status         0.002808         0.004830            0.581           0.561
    L7.User_ID_N                     -0.000001         0.000006           -0.162           0.872
    L7.Title_N                        0.000000         0.000002            0.024           0.981
    L7.Device_Vendor_N               -0.000049         0.000322           -0.151           0.880
    L7.Device_Model_N                 0.000029         0.000038            0.778           0.437
    L7.Content_TV_Show_N             -0.000000         0.000002           -0.104           0.917
    L7.Country_N                      0.000244         0.000489            0.498           0.618
    L7.City_N                         0.000010         0.000009            1.039           0.299
    L7.Region_N                      -0.004444         0.009129           -0.487           0.626
    L8.Playtime                      -0.000000         0.000000           -0.132           0.895
    L8.Interruptions                  0.000003         0.000057            0.058           0.953
    L8.Join Time                     -0.000084         0.000261           -0.320           0.749
    L8.Buffer Ratio                   0.000070         0.000261            0.269           0.788
    L8.Connection Type                0.000144         0.000228            0.633           0.527
    L8.Device                        -0.000284         0.000648           -0.439           0.661
    L8.Device Type                    0.000558         0.000528            1.058           0.290
    L8.Browser                       -0.000088         0.000775           -0.114           0.909
    L8.OS                             0.000884         0.002421            0.365           0.715
    L8.OS Version                    -0.000079         0.000253           -0.312           0.755
    L8.Device ID                     -0.000001         0.000002           -0.610           0.542
    L8.Happiness Score                0.000098         0.000216            0.453           0.651
    L8.Playback Stalls               -0.000468         0.003798           -0.123           0.902
    L8.Startup Error (Count)         -0.000291         0.018235           -0.016           0.987
    L8.Latency                       -0.000000         0.000000           -0.501           0.616
    L8.Crash Status                   0.000342         0.011195            0.031           0.976
    L8.End of Playback Status         0.000253         0.004829            0.052           0.958
    L8.User_ID_N                      0.000001         0.000006            0.169           0.866
    L8.Title_N                       -0.000000         0.000002           -0.051           0.960
    L8.Device_Vendor_N               -0.000042         0.000319           -0.132           0.895
    L8.Device_Model_N                -0.000041         0.000038           -1.102           0.271
    L8.Content_TV_Show_N             -0.000000         0.000002           -0.045           0.964
    L8.Country_N                      0.000746         0.000487            1.532           0.126
    L8.City_N                        -0.000010         0.000009           -1.073           0.283
    L8.Region_N                      -0.000520         0.009092           -0.057           0.954
    L9.Playtime                      -0.000000         0.000000           -0.079           0.937
    L9.Interruptions                  0.000000         0.000057            0.008           0.993
    L9.Join Time                     -0.000570         0.000261           -2.182           0.029
    L9.Buffer Ratio                   0.000235         0.000261            0.901           0.368
    L9.Connection Type               -0.000025         0.000225           -0.110           0.913
    L9.Device                        -0.000135         0.000641           -0.210           0.834
    L9.Device Type                    0.000351         0.000520            0.674           0.500
    L9.Browser                       -0.000479         0.000764           -0.627           0.530
    L9.OS                             0.001525         0.002393            0.637           0.524
    L9.OS Version                    -0.000069         0.000252           -0.273           0.785
    L9.Device ID                     -0.000002         0.000002           -1.404           0.160
    L9.Happiness Score                0.000146         0.000213            0.687           0.492
    L9.Playback Stalls               -0.007771         0.003785           -2.053           0.040
    L9.Startup Error (Count)         -0.003775         0.018218           -0.207           0.836
    L9.Latency                       -0.000000         0.000000           -0.464           0.643
    L9.Crash Status                  -0.003547         0.011186           -0.317           0.751
    L9.End of Playback Status         0.000210         0.004827            0.044           0.965
    L9.User_ID_N                     -0.000004         0.000006           -0.696           0.486
    L9.Title_N                        0.000000         0.000002            0.104           0.917
    L9.Device_Vendor_N                0.000187         0.000315            0.593           0.553
    L9.Device_Model_N                 0.000021         0.000037            0.558           0.577
    L9.Content_TV_Show_N             -0.000002         0.000002           -1.140           0.254
    L9.Country_N                     -0.000244         0.000483           -0.504           0.614
    L9.City_N                        -0.000013         0.000009           -1.482           0.138
    L9.Region_N                       0.006031         0.009034            0.668           0.504
    ============================================================================================
    
    Results for equation Startup Error (Count)
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             0.388938         0.018667           20.835           0.000
    L1.Playtime                      -0.000000         0.000000           -1.202           0.229
    L1.Interruptions                  0.000006         0.000029            0.203           0.839
    L1.Join Time                     -0.000264         0.000133           -1.976           0.048
    L1.Buffer Ratio                  -0.001036         0.000132           -7.834           0.000
    L1.Connection Type                0.000191         0.000114            1.677           0.094
    L1.Device                        -0.000157         0.000326           -0.481           0.631
    L1.Device Type                    0.000642         0.000264            2.428           0.015
    L1.Browser                       -0.000270         0.000388           -0.695           0.487
    L1.OS                             0.001730         0.001217            1.421           0.155
    L1.OS Version                    -0.000256         0.000128           -2.002           0.045
    L1.Device ID                      0.000001         0.000001            0.611           0.541
    L1.Happiness Score               -0.000615         0.000108           -5.678           0.000
    L1.Playback Stalls               -0.001755         0.001924           -0.912           0.362
    L1.Startup Error (Count)          0.046080         0.009267            4.972           0.000
    L1.Latency                       -0.000000         0.000000           -1.459           0.145
    L1.Crash Status                  -0.087943         0.005673          -15.502           0.000
    L1.End of Playback Status        -0.044825         0.002444          -18.341           0.000
    L1.User_ID_N                     -0.000007         0.000003           -2.379           0.017
    L1.Title_N                        0.000001         0.000001            0.955           0.340
    L1.Device_Vendor_N               -0.000260         0.000160           -1.623           0.104
    L1.Device_Model_N                -0.000093         0.000019           -4.950           0.000
    L1.Content_TV_Show_N             -0.000004         0.000001           -4.734           0.000
    L1.Country_N                      0.001016         0.000246            4.139           0.000
    L1.City_N                        -0.000008         0.000005           -1.795           0.073
    L1.Region_N                      -0.006737         0.004588           -1.469           0.142
    L2.Playtime                      -0.000000         0.000000           -1.221           0.222
    L2.Interruptions                  0.000008         0.000029            0.277           0.782
    L2.Join Time                     -0.000245         0.000133           -1.835           0.066
    L2.Buffer Ratio                  -0.000820         0.000133           -6.186           0.000
    L2.Connection Type                0.000071         0.000116            0.612           0.541
    L2.Device                        -0.000631         0.000329           -1.917           0.055
    L2.Device Type                   -0.000290         0.000268           -1.083           0.279
    L2.Browser                        0.000021         0.000393            0.053           0.958
    L2.OS                             0.000286         0.001230            0.233           0.816
    L2.OS Version                     0.000020         0.000129            0.155           0.877
    L2.Device ID                     -0.000001         0.000001           -0.592           0.554
    L2.Happiness Score                0.000049         0.000110            0.446           0.656
    L2.Playback Stalls                0.000080         0.001924            0.042           0.967
    L2.Startup Error (Count)          0.018028         0.009272            1.944           0.052
    L2.Latency                       -0.000000         0.000000           -0.463           0.643
    L2.Crash Status                  -0.004534         0.005687           -0.797           0.425
    L2.End of Playback Status        -0.029036         0.002449          -11.854           0.000
    L2.User_ID_N                      0.000000         0.000003            0.042           0.967
    L2.Title_N                        0.000001         0.000001            0.882           0.378
    L2.Device_Vendor_N                0.000339         0.000162            2.090           0.037
    L2.Device_Model_N                -0.000018         0.000019           -0.941           0.347
    L2.Content_TV_Show_N             -0.000002         0.000001           -2.834           0.005
    L2.Country_N                      0.000730         0.000247            2.953           0.003
    L2.City_N                        -0.000013         0.000005           -2.778           0.005
    L2.Region_N                       0.003459         0.004619            0.749           0.454
    L3.Playtime                      -0.000000         0.000000           -1.779           0.075
    L3.Interruptions                  0.000004         0.000029            0.139           0.890
    L3.Join Time                      0.000110         0.000133            0.821           0.411
    L3.Buffer Ratio                  -0.000057         0.000133           -0.431           0.666
    L3.Connection Type                0.000109         0.000117            0.940           0.347
    L3.Device                         0.000455         0.000331            1.374           0.169
    L3.Device Type                   -0.000449         0.000270           -1.664           0.096
    L3.Browser                       -0.000253         0.000396           -0.641           0.522
    L3.OS                             0.000287         0.001235            0.232           0.816
    L3.OS Version                     0.000034         0.000129            0.263           0.793
    L3.Device ID                      0.000000         0.000001            0.436           0.663
    L3.Happiness Score                0.000142         0.000110            1.287           0.198
    L3.Playback Stalls                0.000127         0.001924            0.066           0.948
    L3.Startup Error (Count)          0.012782         0.009273            1.378           0.168
    L3.Latency                       -0.000000         0.000000           -1.173           0.241
    L3.Crash Status                   0.016752         0.005685            2.947           0.003
    L3.End of Playback Status        -0.012950         0.002452           -5.281           0.000
    L3.User_ID_N                      0.000001         0.000003            0.311           0.756
    L3.Title_N                       -0.000000         0.000001           -0.261           0.794
    L3.Device_Vendor_N                0.000399         0.000163            2.440           0.015
    L3.Device_Model_N                -0.000009         0.000019           -0.488           0.626
    L3.Content_TV_Show_N              0.000001         0.000001            0.953           0.340
    L3.Country_N                      0.000323         0.000249            1.299           0.194
    L3.City_N                        -0.000003         0.000005           -0.675           0.500
    L3.Region_N                      -0.005208         0.004638           -1.123           0.261
    L4.Playtime                      -0.000000         0.000000           -1.119           0.263
    L4.Interruptions                  0.000003         0.000029            0.114           0.909
    L4.Join Time                     -0.000056         0.000133           -0.424           0.671
    L4.Buffer Ratio                  -0.000538         0.000133           -4.057           0.000
    L4.Connection Type               -0.000088         0.000117           -0.754           0.451
    L4.Device                         0.000534         0.000332            1.609           0.108
    L4.Device Type                    0.000389         0.000271            1.436           0.151
    L4.Browser                       -0.000603         0.000397           -1.518           0.129
    L4.OS                             0.000812         0.001238            0.656           0.512
    L4.OS Version                    -0.000012         0.000129           -0.089           0.929
    L4.Device ID                      0.000001         0.000001            1.355           0.175
    L4.Happiness Score                0.000144         0.000110            1.310           0.190
    L4.Playback Stalls                0.001862         0.001923            0.968           0.333
    L4.Startup Error (Count)          0.018867         0.009272            2.035           0.042
    L4.Latency                       -0.000000         0.000000           -2.128           0.033
    L4.Crash Status                   0.024487         0.005685            4.307           0.000
    L4.End of Playback Status        -0.014846         0.002452           -6.053           0.000
    L4.User_ID_N                      0.000000         0.000003            0.111           0.912
    L4.Title_N                        0.000000         0.000001            0.449           0.653
    L4.Device_Vendor_N                0.000035         0.000164            0.215           0.830
    L4.Device_Model_N                -0.000011         0.000019           -0.572           0.567
    L4.Content_TV_Show_N             -0.000000         0.000001           -0.447           0.655
    L4.Country_N                     -0.000038         0.000249           -0.151           0.880
    L4.City_N                        -0.000003         0.000005           -0.631           0.528
    L4.Region_N                      -0.004210         0.004651           -0.905           0.365
    L5.Playtime                       0.000000         0.000000            0.566           0.571
    L5.Interruptions                 -0.000002         0.000029           -0.085           0.932
    L5.Join Time                      0.000328         0.000133            2.472           0.013
    L5.Buffer Ratio                  -0.000285         0.000132           -2.148           0.032
    L5.Connection Type               -0.000190         0.000117           -1.621           0.105
    L5.Device                        -0.000084         0.000332           -0.254           0.800
    L5.Device Type                   -0.000552         0.000271           -2.033           0.042
    L5.Browser                       -0.000009         0.000398           -0.022           0.983
    L5.OS                            -0.000778         0.001239           -0.628           0.530
    L5.OS Version                     0.000093         0.000130            0.718           0.473
    L5.Device ID                     -0.000000         0.000001           -0.326           0.745
    L5.Happiness Score                0.000194         0.000110            1.756           0.079
    L5.Playback Stalls                0.000375         0.001873            0.200           0.841
    L5.Startup Error (Count)          0.004110         0.009271            0.443           0.658
    L5.Latency                       -0.000000         0.000000           -0.859           0.390
    L5.Crash Status                   0.026925         0.005686            4.735           0.000
    L5.End of Playback Status        -0.009746         0.002453           -3.972           0.000
    L5.User_ID_N                      0.000003         0.000003            0.913           0.361
    L5.Title_N                       -0.000000         0.000001           -0.135           0.893
    L5.Device_Vendor_N               -0.000052         0.000164           -0.320           0.749
    L5.Device_Model_N                -0.000026         0.000019           -1.338           0.181
    L5.Content_TV_Show_N              0.000001         0.000001            1.627           0.104
    L5.Country_N                      0.000272         0.000250            1.090           0.276
    L5.City_N                         0.000003         0.000005            0.539           0.590
    L5.Region_N                       0.001081         0.004657            0.232           0.816
    L6.Playtime                       0.000000         0.000000            0.379           0.705
    L6.Interruptions                 -0.000004         0.000029           -0.138           0.890
    L6.Join Time                      0.000193         0.000133            1.453           0.146
    L6.Buffer Ratio                  -0.000165         0.000132           -1.246           0.213
    L6.Connection Type                0.000004         0.000117            0.035           0.972
    L6.Device                         0.000182         0.000332            0.548           0.583
    L6.Device Type                   -0.000635         0.000271           -2.341           0.019
    L6.Browser                        0.000645         0.000397            1.626           0.104
    L6.OS                            -0.003639         0.001238           -2.940           0.003
    L6.OS Version                     0.000304         0.000129            2.349           0.019
    L6.Device ID                     -0.000000         0.000001           -0.425           0.671
    L6.Happiness Score                0.000106         0.000110            0.958           0.338
    L6.Playback Stalls                0.000263         0.001919            0.137           0.891
    L6.Startup Error (Count)          0.029122         0.009270            3.142           0.002
    L6.Latency                        0.000000         0.000000            1.521           0.128
    L6.Crash Status                   0.006300         0.005686            1.108           0.268
    L6.End of Playback Status        -0.006684         0.002454           -2.724           0.006
    L6.User_ID_N                     -0.000004         0.000003           -1.269           0.204
    L6.Title_N                       -0.000000         0.000001           -0.276           0.782
    L6.Device_Vendor_N                0.000071         0.000164            0.433           0.665
    L6.Device_Model_N                 0.000029         0.000019            1.520           0.129
    L6.Content_TV_Show_N              0.000000         0.000001            0.311           0.755
    L6.Country_N                      0.000192         0.000249            0.768           0.442
    L6.City_N                         0.000006         0.000005            1.228           0.220
    L6.Region_N                      -0.009674         0.004650           -2.080           0.038
    L7.Playtime                       0.000000         0.000000            0.224           0.823
    L7.Interruptions                 -0.000003         0.000029           -0.101           0.920
    L7.Join Time                     -0.000163         0.000133           -1.225           0.221
    L7.Buffer Ratio                  -0.000199         0.000132           -1.500           0.134
    L7.Connection Type                0.000052         0.000117            0.445           0.657
    L7.Device                         0.000079         0.000331            0.240           0.810
    L7.Device Type                   -0.000217         0.000270           -0.804           0.421
    L7.Browser                       -0.000411         0.000396           -1.039           0.299
    L7.OS                             0.001477         0.001234            1.197           0.231
    L7.OS Version                    -0.000145         0.000129           -1.127           0.260
    L7.Device ID                     -0.000002         0.000001           -1.866           0.062
    L7.Happiness Score               -0.000118         0.000110           -1.072           0.284
    L7.Playback Stalls                0.001125         0.001929            0.583           0.560
    L7.Startup Error (Count)          0.014574         0.009269            1.572           0.116
    L7.Latency                       -0.000000         0.000000           -0.255           0.798
    L7.Crash Status                   0.015019         0.005686            2.642           0.008
    L7.End of Playback Status        -0.002969         0.002453           -1.210           0.226
    L7.User_ID_N                     -0.000001         0.000003           -0.262           0.793
    L7.Title_N                       -0.000000         0.000001           -0.630           0.529
    L7.Device_Vendor_N                0.000356         0.000163            2.178           0.029
    L7.Device_Model_N                -0.000007         0.000019           -0.385           0.700
    L7.Content_TV_Show_N             -0.000001         0.000001           -1.425           0.154
    L7.Country_N                     -0.000467         0.000249           -1.880           0.060
    L7.City_N                         0.000001         0.000005            0.264           0.792
    L7.Region_N                       0.005992         0.004637            1.292           0.196
    L8.Playtime                      -0.000000         0.000000           -0.336           0.737
    L8.Interruptions                  0.000001         0.000029            0.032           0.975
    L8.Join Time                     -0.000082         0.000133           -0.621           0.535
    L8.Buffer Ratio                   0.000005         0.000132            0.034           0.972
    L8.Connection Type                0.000061         0.000116            0.530           0.596
    L8.Device                         0.000020         0.000329            0.060           0.953
    L8.Device Type                    0.000142         0.000268            0.529           0.597
    L8.Browser                        0.000056         0.000394            0.141           0.888
    L8.OS                            -0.000409         0.001230           -0.332           0.740
    L8.OS Version                    -0.000029         0.000129           -0.223           0.823
    L8.Device ID                      0.000000         0.000001            0.418           0.676
    L8.Happiness Score               -0.000053         0.000110           -0.486           0.627
    L8.Playback Stalls                0.000897         0.001929            0.465           0.642
    L8.Startup Error (Count)          0.043725         0.009263            4.721           0.000
    L8.Latency                       -0.000000         0.000000           -1.691           0.091
    L8.Crash Status                  -0.002593         0.005687           -0.456           0.648
    L8.End of Playback Status         0.005142         0.002453            2.096           0.036
    L8.User_ID_N                     -0.000003         0.000003           -0.994           0.320
    L8.Title_N                       -0.000001         0.000001           -0.707           0.479
    L8.Device_Vendor_N               -0.000192         0.000162           -1.183           0.237
    L8.Device_Model_N                 0.000001         0.000019            0.029           0.977
    L8.Content_TV_Show_N             -0.000003         0.000001           -3.657           0.000
    L8.Country_N                      0.000101         0.000247            0.410           0.682
    L8.City_N                         0.000007         0.000005            1.473           0.141
    L8.Region_N                      -0.005566         0.004618           -1.205           0.228
    L9.Playtime                       0.000000         0.000000            0.263           0.792
    L9.Interruptions                 -0.000006         0.000029           -0.208           0.835
    L9.Join Time                     -0.000064         0.000133           -0.480           0.631
    L9.Buffer Ratio                  -0.000166         0.000132           -1.256           0.209
    L9.Connection Type                0.000058         0.000114            0.505           0.614
    L9.Device                        -0.000167         0.000326           -0.514           0.607
    L9.Device Type                   -0.000405         0.000264           -1.531           0.126
    L9.Browser                        0.000928         0.000388            2.392           0.017
    L9.OS                            -0.001944         0.001216           -1.599           0.110
    L9.OS Version                     0.000088         0.000128            0.685           0.493
    L9.Device ID                      0.000000         0.000001            0.122           0.903
    L9.Happiness Score               -0.000100         0.000108           -0.920           0.358
    L9.Playback Stalls                0.000114         0.001923            0.059           0.953
    L9.Startup Error (Count)          0.027147         0.009254            2.933           0.003
    L9.Latency                        0.000000         0.000000            0.164           0.870
    L9.Crash Status                   0.006253         0.005682            1.100           0.271
    L9.End of Playback Status        -0.001396         0.002452           -0.569           0.569
    L9.User_ID_N                     -0.000001         0.000003           -0.312           0.755
    L9.Title_N                        0.000000         0.000001            0.123           0.902
    L9.Device_Vendor_N                0.000029         0.000160            0.183           0.855
    L9.Device_Model_N                 0.000014         0.000019            0.752           0.452
    L9.Content_TV_Show_N             -0.000002         0.000001           -2.777           0.005
    L9.Country_N                      0.000366         0.000246            1.491           0.136
    L9.City_N                         0.000004         0.000005            0.889           0.374
    L9.Region_N                      -0.005964         0.004589           -1.300           0.194
    ============================================================================================
    
    Results for equation Latency
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                         16198.276948      4083.657881            3.967           0.000
    L1.Playtime                       0.178203         0.049793            3.579           0.000
    L1.Interruptions                 -6.210503         6.389656           -0.972           0.331
    L1.Join Time                     29.045122        29.173733            0.996           0.319
    L1.Buffer Ratio                  41.429952        28.942654            1.431           0.152
    L1.Connection Type               46.669514        24.967341            1.869           0.062
    L1.Device                      -149.605066        71.293404           -2.098           0.036
    L1.Device Type                 -112.452018        57.836244           -1.944           0.052
    L1.Browser                      -27.802412        84.842943           -0.328           0.743
    L1.OS                          -326.968973       266.210864           -1.228           0.219
    L1.OS Version                    41.030522        27.965987            1.467           0.142
    L1.Device ID                     -1.305654         0.195578           -6.676           0.000
    L1.Happiness Score              331.975859        23.710514           14.001           0.000
    L1.Playback Stalls             -119.131085       420.878649           -0.283           0.777
    L1.Startup Error (Count)       3138.295858      2027.285204            1.548           0.122
    L1.Latency                        0.014727         0.003661            4.022           0.000
    L1.Crash Status                -570.444438      1241.027506           -0.460           0.646
    L1.End of Playback Status       944.823245       534.658467            1.767           0.077
    L1.User_ID_N                      0.445459         0.621858            0.716           0.474
    L1.Title_N                       -1.235448         0.168655           -7.325           0.000
    L1.Device_Vendor_N             -110.579326        35.019846           -3.158           0.002
    L1.Device_Model_N                 7.399462         4.129046            1.792           0.073
    L1.Content_TV_Show_N              0.274805         0.167157            1.644           0.100
    L1.Country_N                    142.735931        53.716459            2.657           0.008
    L1.City_N                        -3.608501         0.996973           -3.619           0.000
    L1.Region_N                    1130.255949      1003.572069            1.126           0.260
    L2.Playtime                       0.007044         0.049799            0.141           0.888
    L2.Interruptions                 -0.154097         6.388293           -0.024           0.981
    L2.Join Time                      5.445144        29.182085            0.187           0.852
    L2.Buffer Ratio                 -14.839847        29.011580           -0.512           0.609
    L2.Connection Type               76.066965        25.283617            3.009           0.003
    L2.Device                       -66.766071        71.986712           -0.927           0.354
    L2.Device Type                   74.235123        58.620657            1.266           0.205
    L2.Browser                     -241.403845        86.059219           -2.805           0.005
    L2.OS                           312.599840       269.114827            1.162           0.245
    L2.OS Version                   -13.159690        28.159910           -0.467           0.640
    L2.Device ID                     -0.408996         0.197901           -2.067           0.039
    L2.Happiness Score              179.902228        24.022971            7.489           0.000
    L2.Playback Stalls              170.275595       420.907495            0.405           0.686
    L2.Startup Error (Count)      -1745.158743      2028.449693           -0.860           0.390
    L2.Latency                        0.013650         0.003662            3.728           0.000
    L2.Crash Status                 -98.159550      1244.113335           -0.079           0.937
    L2.End of Playback Status      -886.965806       535.833130           -1.655           0.098
    L2.User_ID_N                      0.564504         0.631466            0.894           0.371
    L2.Title_N                       -0.685406         0.168905           -4.058           0.000
    L2.Device_Vendor_N              -32.668997        35.474828           -0.921           0.357
    L2.Device_Model_N                 2.424421         4.178324            0.580           0.562
    L2.Content_TV_Show_N             -0.142633         0.167843           -0.850           0.395
    L2.Country_N                    -44.517054        54.101871           -0.823           0.411
    L2.City_N                        -0.877319         1.012214           -0.867           0.386
    L2.Region_N                    1518.932916      1010.420598            1.503           0.133
    L3.Playtime                      -0.062366         0.049798           -1.252           0.210
    L3.Interruptions                  1.562594         6.388170            0.245           0.807
    L3.Join Time                    -26.237059        29.181097           -0.899           0.369
    L3.Buffer Ratio                  12.973059        29.022963            0.447           0.655
    L3.Connection Type               33.452864        25.489154            1.312           0.189
    L3.Device                        15.150216        72.354745            0.209           0.834
    L3.Device Type                   84.283557        59.070290            1.427           0.154
    L3.Browser                     -227.557713        86.542786           -2.629           0.009
    L3.OS                           191.375647       270.076434            0.709           0.479
    L3.OS Version                    -0.297424        28.233752           -0.011           0.992
    L3.Device ID                      0.026780         0.199400            0.134           0.893
    L3.Happiness Score              151.506195        24.096122            6.288           0.000
    L3.Playback Stalls             -471.759455       420.930781           -1.121           0.262
    L3.Startup Error (Count)        -51.428025      2028.484263           -0.025           0.980
    L3.Latency                        0.012170         0.003661            3.324           0.001
    L3.Crash Status                -925.180335      1243.749241           -0.744           0.457
    L3.End of Playback Status      -404.959258       536.387893           -0.755           0.450
    L3.User_ID_N                     -0.131558         0.636080           -0.207           0.836
    L3.Title_N                       -0.251926         0.169008           -1.491           0.136
    L3.Device_Vendor_N              -41.681516        35.724487           -1.167           0.243
    L3.Device_Model_N                -2.953340         4.200197           -0.703           0.482
    L3.Content_TV_Show_N             -0.433257         0.168239           -2.575           0.010
    L3.Country_N                    -78.642415        54.412241           -1.445           0.148
    L3.City_N                         1.628379         1.019903            1.597           0.110
    L3.Region_N                     365.024684      1014.552220            0.360           0.719
    L4.Playtime                      -0.039362         0.049805           -0.790           0.429
    L4.Interruptions                 -1.678328         6.388152           -0.263           0.793
    L4.Join Time                      1.074260        29.025743            0.037           0.970
    L4.Buffer Ratio                  25.800188        29.020940            0.889           0.374
    L4.Connection Type               14.624312        25.590176            0.571           0.568
    L4.Device                        29.862817        72.522235            0.412           0.681
    L4.Device Type                  -84.824503        59.280417           -1.431           0.152
    L4.Browser                       23.181800        86.838777            0.267           0.790
    L4.OS                          -206.914790       270.811896           -0.764           0.445
    L4.OS Version                    10.583643        28.306557            0.374           0.708
    L4.Device ID                     -0.129991         0.200055           -0.650           0.516
    L4.Happiness Score               83.696798        24.120246            3.470           0.001
    L4.Playback Stalls              -82.374463       420.761804           -0.196           0.845
    L4.Startup Error (Count)       3555.746771      2028.384096            1.753           0.080
    L4.Latency                        0.012075         0.003662            3.298           0.001
    L4.Crash Status               -3652.697222      1243.743597           -2.937           0.003
    L4.End of Playback Status       579.661904       536.503135            1.080           0.280
    L4.User_ID_N                      0.000287         0.638126            0.000           1.000
    L4.Title_N                        0.143235         0.168946            0.848           0.397
    L4.Device_Vendor_N               38.604698        35.847678            1.077           0.282
    L4.Device_Model_N                 4.092246         4.210710            0.972           0.331
    L4.Content_TV_Show_N             -0.209048         0.168408           -1.241           0.214
    L4.Country_N                    -11.498672        54.549846           -0.211           0.833
    L4.City_N                        -0.748295         1.023090           -0.731           0.465
    L4.Region_N                     854.313209      1017.425638            0.840           0.401
    L5.Playtime                       0.055043         0.049811            1.105           0.269
    L5.Interruptions                 -1.110823         6.388345           -0.174           0.862
    L5.Join Time                    -40.099621        29.036838           -1.381           0.167
    L5.Buffer Ratio                  -8.824274        28.978864           -0.305           0.761
    L5.Connection Type               -4.317296        25.611660           -0.169           0.866
    L5.Device                       109.320776        72.573133            1.506           0.132
    L5.Device Type                   68.522176        59.352936            1.154           0.248
    L5.Browser                      -12.727023        86.973833           -0.146           0.884
    L5.OS                           -66.864851       271.082135           -0.247           0.805
    L5.OS Version                   -20.788974        28.335227           -0.734           0.463
    L5.Device ID                     -0.234644         0.200178           -1.172           0.241
    L5.Happiness Score               77.044626        24.123367            3.194           0.001
    L5.Playback Stalls              283.298306       409.719524            0.691           0.489
    L5.Startup Error (Count)       -335.970001      2028.193522           -0.166           0.868
    L5.Latency                        0.006960         0.003662            1.901           0.057
    L5.Crash Status                 194.180627      1243.912049            0.156           0.876
    L5.End of Playback Status      -111.553434       536.714201           -0.208           0.835
    L5.User_ID_N                      0.074254         0.638709            0.116           0.907
    L5.Title_N                        0.395524         0.168959            2.341           0.019
    L5.Device_Vendor_N               14.858219        35.873346            0.414           0.679
    L5.Device_Model_N                -1.440100         4.213366           -0.342           0.733
    L5.Content_TV_Show_N             -0.310699         0.168503           -1.844           0.065
    L5.Country_N                    -25.514547        54.651337           -0.467           0.641
    L5.City_N                        -0.962195         1.024451           -0.939           0.348
    L5.Region_N                     633.313333      1018.806052            0.622           0.534
    L6.Playtime                      -0.019968         0.049812           -0.401           0.689
    L6.Interruptions                 -1.925646         6.388342           -0.301           0.763
    L6.Join Time                     28.736889        29.048404            0.989           0.323
    L6.Buffer Ratio                 -22.072196        28.980039           -0.762           0.446
    L6.Connection Type              -21.206489        25.590400           -0.829           0.407
    L6.Device                        30.481145        72.528971            0.420           0.674
    L6.Device Type                   63.692135        59.297169            1.074           0.283
    L6.Browser                      106.824308        86.834443            1.230           0.219
    L6.OS                            88.583774       270.813402            0.327           0.744
    L6.OS Version                   -29.505607        28.308785           -1.042           0.297
    L6.Device ID                      0.075141         0.200037            0.376           0.707
    L6.Happiness Score               35.618512        24.105893            1.478           0.140
    L6.Playback Stalls              -94.946770       419.863507           -0.226           0.821
    L6.Startup Error (Count)      -1617.177239      2027.909827           -0.797           0.425
    L6.Latency                        0.012264         0.003662            3.350           0.001
    L6.Crash Status                 991.615766      1243.822426            0.797           0.425
    L6.End of Playback Status        90.774879       536.792019            0.169           0.866
    L6.User_ID_N                     -0.721707         0.638140           -1.131           0.258
    L6.Title_N                        0.001958         0.169000            0.012           0.991
    L6.Device_Vendor_N               54.991636        35.843793            1.534           0.125
    L6.Device_Model_N                 4.060738         4.209950            0.965           0.335
    L6.Content_TV_Show_N              0.119278         0.168448            0.708           0.479
    L6.Country_N                    -12.584948        54.540835           -0.231           0.818
    L6.City_N                        -0.342719         1.023142           -0.335           0.738
    L6.Region_N                     477.868841      1017.289504            0.470           0.639
    L7.Playtime                      -0.013313         0.049807           -0.267           0.789
    L7.Interruptions                  0.196246         6.388595            0.031           0.975
    L7.Join Time                     31.356315        29.044948            1.080           0.280
    L7.Buffer Ratio                  -9.094099        28.984907           -0.314           0.754
    L7.Connection Type               15.844933        25.488588            0.622           0.534
    L7.Device                         7.442219        72.333032            0.103           0.918
    L7.Device Type                   62.348658        59.072064            1.055           0.291
    L7.Browser                     -152.359807        86.556268           -1.760           0.078
    L7.OS                            -8.934096       270.057084           -0.033           0.974
    L7.OS Version                    25.073934        28.238683            0.888           0.375
    L7.Device ID                     -0.320461         0.199382           -1.607           0.108
    L7.Happiness Score              -26.479140        24.073423           -1.100           0.271
    L7.Playback Stalls             -199.189454       422.046261           -0.472           0.637
    L7.Startup Error (Count)      -1103.554388      2027.737765           -0.544           0.586
    L7.Latency                        0.025886         0.003662            7.070           0.000
    L7.Crash Status                  98.748281      1243.799838            0.079           0.937
    L7.End of Playback Status        80.835260       536.718213            0.151           0.880
    L7.User_ID_N                      0.230454         0.636079            0.362           0.717
    L7.Title_N                        0.080577         0.169025            0.477           0.634
    L7.Device_Vendor_N                7.397344        35.729180            0.207           0.836
    L7.Device_Model_N                 0.298701         4.199439            0.071           0.943
    L7.Content_TV_Show_N             -0.004444         0.168252           -0.026           0.979
    L7.Country_N                    -91.491429        54.394269           -1.682           0.093
    L7.City_N                         1.210683         1.019916            1.187           0.235
    L7.Region_N                    1751.564221      1014.495570            1.727           0.084
    L8.Playtime                      -0.082639         0.049804           -1.659           0.097
    L8.Interruptions                 -1.685348         6.388430           -0.264           0.792
    L8.Join Time                      5.672372        29.042007            0.195           0.845
    L8.Buffer Ratio                  -8.335691        28.980348           -0.288           0.774
    L8.Connection Type               20.408296        25.286017            0.807           0.420
    L8.Device                       -35.563307        71.964000           -0.494           0.621
    L8.Device Type                   70.053655        58.627315            1.195           0.232
    L8.Browser                     -104.408045        86.096558           -1.213           0.225
    L8.OS                           427.188805       269.060072            1.588           0.112
    L8.OS Version                   -33.968088        28.155428           -1.206           0.228
    L8.Device ID                      0.223597         0.197878            1.130           0.258
    L8.Happiness Score               22.937114        24.008612            0.955           0.339
    L8.Playback Stalls              113.144533       422.055232            0.268           0.789
    L8.Startup Error (Count)      -3393.667544      2026.295359           -1.675           0.094
    L8.Latency                        0.018427         0.003662            5.032           0.000
    L8.Crash Status                1837.298021      1244.035118            1.477           0.140
    L8.End of Playback Status      -825.278255       536.586477           -1.538           0.124
    L8.User_ID_N                     -0.821539         0.631490           -1.301           0.193
    L8.Title_N                       -0.071384         0.168872           -0.423           0.673
    L8.Device_Vendor_N              -67.075812        35.477372           -1.891           0.059
    L8.Device_Model_N                -5.646639         4.177689           -1.352           0.176
    L8.Content_TV_Show_N              0.048096         0.167800            0.287           0.774
    L8.Country_N                   -104.776354        54.085401           -1.937           0.053
    L8.City_N                         0.430939         1.012049            0.426           0.670
    L8.Region_N                    1233.830347      1010.312550            1.221           0.222
    L9.Playtime                      -0.008986         0.049792           -0.180           0.857
    L9.Interruptions                  1.390273         6.388464            0.218           0.828
    L9.Join Time                     21.984912        29.025742            0.757           0.449
    L9.Buffer Ratio                 -36.577994        28.974394           -1.262           0.207
    L9.Connection Type                9.436692        24.969744            0.378           0.705
    L9.Device                       -43.694663        71.247842           -0.613           0.540
    L9.Device Type                    5.744305        57.804581            0.099           0.921
    L9.Browser                       -8.183445        84.856981           -0.096           0.923
    L9.OS                           205.844709       265.919942            0.774           0.439
    L9.OS Version                   -42.503076        27.956414           -1.520           0.128
    L9.Device ID                     -0.391149         0.195711           -1.999           0.046
    L9.Happiness Score                2.564768        23.682786            0.108           0.914
    L9.Playback Stalls              126.389666       420.592262            0.301           0.764
    L9.Startup Error (Count)      -1914.715670      2024.469207           -0.946           0.344
    L9.Latency                        0.016139         0.003659            4.411           0.000
    L9.Crash Status                 526.200556      1242.998565            0.423           0.672
    L9.End of Playback Status      -512.341655       536.393493           -0.955           0.339
    L9.User_ID_N                     -0.296214         0.622198           -0.476           0.634
    L9.Title_N                       -0.314145         0.168584           -1.863           0.062
    L9.Device_Vendor_N               -2.922630        35.020068           -0.083           0.933
    L9.Device_Model_N                -0.151689         4.128980           -0.037           0.971
    L9.Content_TV_Show_N             -0.007139         0.167087           -0.043           0.966
    L9.Country_N                    -62.007697        53.708090           -1.155           0.248
    L9.City_N                        -0.421096         0.997813           -0.422           0.673
    L9.Region_N                    1960.304606      1003.842827            1.953           0.051
    ============================================================================================
    
    Results for equation Crash Status
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             1.130393         0.022512           50.212           0.000
    L1.Playtime                      -0.000000         0.000000           -1.110           0.267
    L1.Interruptions                  0.000007         0.000035            0.207           0.836
    L1.Join Time                     -0.000504         0.000161           -3.135           0.002
    L1.Buffer Ratio                  -0.001185         0.000160           -7.426           0.000
    L1.Connection Type                0.000056         0.000138            0.407           0.684
    L1.Device                        -0.000387         0.000393           -0.985           0.325
    L1.Device Type                    0.000598         0.000319            1.875           0.061
    L1.Browser                       -0.001031         0.000468           -2.204           0.028
    L1.OS                             0.002809         0.001468            1.914           0.056
    L1.OS Version                    -0.000162         0.000154           -1.051           0.293
    L1.Device ID                      0.000000         0.000001            0.234           0.815
    L1.Happiness Score               -0.000666         0.000131           -5.092           0.000
    L1.Playback Stalls               -0.003205         0.002320           -1.382           0.167
    L1.Startup Error (Count)          0.037261         0.011176            3.334           0.001
    L1.Latency                       -0.000000         0.000000           -1.727           0.084
    L1.Crash Status                  -0.098496         0.006841          -14.397           0.000
    L1.End of Playback Status        -0.041784         0.002947          -14.176           0.000
    L1.User_ID_N                     -0.000009         0.000003           -2.535           0.011
    L1.Title_N                        0.000001         0.000001            0.677           0.498
    L1.Device_Vendor_N               -0.000117         0.000193           -0.606           0.544
    L1.Device_Model_N                -0.000107         0.000023           -4.706           0.000
    L1.Content_TV_Show_N             -0.000003         0.000001           -2.933           0.003
    L1.Country_N                      0.000540         0.000296            1.823           0.068
    L1.City_N                         0.000001         0.000005            0.169           0.866
    L1.Region_N                      -0.005188         0.005532           -0.938           0.348
    L2.Playtime                      -0.000000         0.000000           -1.323           0.186
    L2.Interruptions                  0.000002         0.000035            0.047           0.963
    L2.Join Time                     -0.000164         0.000161           -1.017           0.309
    L2.Buffer Ratio                  -0.000829         0.000160           -5.181           0.000
    L2.Connection Type                0.000236         0.000139            1.694           0.090
    L2.Device                        -0.000456         0.000397           -1.149           0.251
    L2.Device Type                   -0.000250         0.000323           -0.773           0.439
    L2.Browser                       -0.000481         0.000474           -1.013           0.311
    L2.OS                             0.000810         0.001484            0.546           0.585
    L2.OS Version                     0.000024         0.000155            0.157           0.875
    L2.Device ID                      0.000000         0.000001            0.061           0.952
    L2.Happiness Score               -0.000095         0.000132           -0.718           0.473
    L2.Playback Stalls                0.001099         0.002320            0.474           0.636
    L2.Startup Error (Count)          0.011897         0.011182            1.064           0.287
    L2.Latency                        0.000000         0.000000            0.493           0.622
    L2.Crash Status                  -0.006479         0.006858           -0.945           0.345
    L2.End of Playback Status        -0.019620         0.002954           -6.642           0.000
    L2.User_ID_N                     -0.000002         0.000003           -0.522           0.602
    L2.Title_N                        0.000002         0.000001            1.857           0.063
    L2.Device_Vendor_N                0.000351         0.000196            1.795           0.073
    L2.Device_Model_N                -0.000074         0.000023           -3.206           0.001
    L2.Content_TV_Show_N             -0.000002         0.000001           -2.373           0.018
    L2.Country_N                      0.000423         0.000298            1.418           0.156
    L2.City_N                        -0.000009         0.000006           -1.657           0.097
    L2.Region_N                       0.007135         0.005570            1.281           0.200
    L3.Playtime                      -0.000000         0.000000           -0.966           0.334
    L3.Interruptions                  0.000003         0.000035            0.095           0.924
    L3.Join Time                     -0.000175         0.000161           -1.090           0.276
    L3.Buffer Ratio                   0.000155         0.000160            0.972           0.331
    L3.Connection Type                0.000163         0.000141            1.162           0.245
    L3.Device                         0.000279         0.000399            0.699           0.484
    L3.Device Type                    0.000011         0.000326            0.035           0.972
    L3.Browser                       -0.000351         0.000477           -0.735           0.462
    L3.OS                             0.002186         0.001489            1.468           0.142
    L3.OS Version                    -0.000214         0.000156           -1.375           0.169
    L3.Device ID                     -0.000000         0.000001           -0.428           0.669
    L3.Happiness Score                0.000055         0.000133            0.418           0.676
    L3.Playback Stalls                0.000974         0.002320            0.420           0.675
    L3.Startup Error (Count)          0.009119         0.011183            0.815           0.415
    L3.Latency                       -0.000000         0.000000           -1.172           0.241
    L3.Crash Status                   0.029404         0.006856            4.288           0.000
    L3.End of Playback Status        -0.004536         0.002957           -1.534           0.125
    L3.User_ID_N                     -0.000006         0.000004           -1.853           0.064
    L3.Title_N                        0.000000         0.000001            0.080           0.936
    L3.Device_Vendor_N                0.000407         0.000197            2.065           0.039
    L3.Device_Model_N                -0.000003         0.000023           -0.141           0.888
    L3.Content_TV_Show_N             -0.000000         0.000001           -0.161           0.872
    L3.Country_N                      0.000385         0.000300            1.283           0.199
    L3.City_N                        -0.000006         0.000006           -0.986           0.324
    L3.Region_N                       0.001462         0.005593            0.261           0.794
    L4.Playtime                      -0.000000         0.000000           -0.702           0.483
    L4.Interruptions                  0.000005         0.000035            0.155           0.877
    L4.Join Time                     -0.000284         0.000160           -1.775           0.076
    L4.Buffer Ratio                  -0.000213         0.000160           -1.334           0.182
    L4.Connection Type               -0.000044         0.000141           -0.314           0.753
    L4.Device                         0.000906         0.000400            2.267           0.023
    L4.Device Type                    0.000276         0.000327            0.843           0.399
    L4.Browser                       -0.000153         0.000479           -0.319           0.750
    L4.OS                             0.000351         0.001493            0.235           0.814
    L4.OS Version                    -0.000093         0.000156           -0.598           0.550
    L4.Device ID                      0.000002         0.000001            2.047           0.041
    L4.Happiness Score                0.000133         0.000133            1.000           0.317
    L4.Playback Stalls               -0.002343         0.002320           -1.010           0.312
    L4.Startup Error (Count)          0.018433         0.011182            1.648           0.099
    L4.Latency                       -0.000000         0.000000           -2.203           0.028
    L4.Crash Status                   0.041828         0.006856            6.100           0.000
    L4.End of Playback Status        -0.003027         0.002958           -1.023           0.306
    L4.User_ID_N                      0.000001         0.000004            0.327           0.744
    L4.Title_N                        0.000001         0.000001            0.654           0.513
    L4.Device_Vendor_N               -0.000129         0.000198           -0.652           0.514
    L4.Device_Model_N                -0.000032         0.000023           -1.368           0.171
    L4.Content_TV_Show_N              0.000001         0.000001            0.723           0.469
    L4.Country_N                     -0.000241         0.000301           -0.800           0.424
    L4.City_N                        -0.000003         0.000006           -0.592           0.554
    L4.Region_N                      -0.001667         0.005609           -0.297           0.766
    L5.Playtime                       0.000000         0.000000            0.660           0.510
    L5.Interruptions                 -0.000003         0.000035           -0.082           0.934
    L5.Join Time                      0.000077         0.000160            0.483           0.629
    L5.Buffer Ratio                  -0.000271         0.000160           -1.694           0.090
    L5.Connection Type               -0.000115         0.000141           -0.817           0.414
    L5.Device                        -0.000408         0.000400           -1.019           0.308
    L5.Device Type                   -0.000639         0.000327           -1.952           0.051
    L5.Browser                       -0.000135         0.000479           -0.282           0.778
    L5.OS                            -0.001078         0.001494           -0.722           0.471
    L5.OS Version                     0.000225         0.000156            1.437           0.151
    L5.Device ID                     -0.000001         0.000001           -0.989           0.323
    L5.Happiness Score                0.000264         0.000133            1.983           0.047
    L5.Playback Stalls               -0.002966         0.002259           -1.313           0.189
    L5.Startup Error (Count)         -0.014561         0.011181           -1.302           0.193
    L5.Latency                       -0.000000         0.000000           -0.202           0.840
    L5.Crash Status                   0.048591         0.006857            7.086           0.000
    L5.End of Playback Status        -0.006142         0.002959           -2.076           0.038
    L5.User_ID_N                      0.000006         0.000004            1.608           0.108
    L5.Title_N                       -0.000001         0.000001           -0.988           0.323
    L5.Device_Vendor_N                0.000076         0.000198            0.384           0.701
    L5.Device_Model_N                -0.000014         0.000023           -0.623           0.533
    L5.Content_TV_Show_N              0.000001         0.000001            0.885           0.376
    L5.Country_N                      0.000190         0.000301            0.630           0.529
    L5.City_N                         0.000001         0.000006            0.150           0.880
    L5.Region_N                       0.001090         0.005616            0.194           0.846
    L6.Playtime                       0.000000         0.000000            0.241           0.809
    L6.Interruptions                 -0.000004         0.000035           -0.120           0.905
    L6.Join Time                      0.000274         0.000160            1.711           0.087
    L6.Buffer Ratio                  -0.000002         0.000160           -0.014           0.989
    L6.Connection Type               -0.000006         0.000141           -0.042           0.966
    L6.Device                        -0.000313         0.000400           -0.783           0.433
    L6.Device Type                   -0.000404         0.000327           -1.237           0.216
    L6.Browser                        0.000604         0.000479            1.261           0.207
    L6.OS                            -0.001706         0.001493           -1.143           0.253
    L6.OS Version                     0.000107         0.000156            0.685           0.494
    L6.Device ID                     -0.000001         0.000001           -1.116           0.264
    L6.Happiness Score                0.000025         0.000133            0.187           0.852
    L6.Playback Stalls                0.002102         0.002315            0.908           0.364
    L6.Startup Error (Count)          0.014645         0.011179            1.310           0.190
    L6.Latency                        0.000000         0.000000            2.320           0.020
    L6.Crash Status                   0.026248         0.006857            3.828           0.000
    L6.End of Playback Status        -0.004276         0.002959           -1.445           0.148
    L6.User_ID_N                     -0.000001         0.000004           -0.156           0.876
    L6.Title_N                        0.000000         0.000001            0.113           0.910
    L6.Device_Vendor_N               -0.000172         0.000198           -0.871           0.384
    L6.Device_Model_N                 0.000041         0.000023            1.783           0.075
    L6.Content_TV_Show_N              0.000001         0.000001            0.756           0.450
    L6.Country_N                     -0.000292         0.000301           -0.972           0.331
    L6.City_N                         0.000008         0.000006            1.353           0.176
    L6.Region_N                      -0.001214         0.005608           -0.216           0.829
    L7.Playtime                       0.000000         0.000000            0.223           0.824
    L7.Interruptions                 -0.000000         0.000035           -0.005           0.996
    L7.Join Time                     -0.000150         0.000160           -0.934           0.350
    L7.Buffer Ratio                  -0.000180         0.000160           -1.127           0.260
    L7.Connection Type                0.000054         0.000141            0.386           0.700
    L7.Device                         0.000058         0.000399            0.144           0.885
    L7.Device Type                    0.000320         0.000326            0.981           0.326
    L7.Browser                       -0.000136         0.000477           -0.286           0.775
    L7.OS                             0.002415         0.001489            1.622           0.105
    L7.OS Version                    -0.000246         0.000156           -1.578           0.115
    L7.Device ID                     -0.000001         0.000001           -1.206           0.228
    L7.Happiness Score                0.000018         0.000133            0.137           0.891
    L7.Playback Stalls                0.002823         0.002327            1.213           0.225
    L7.Startup Error (Count)         -0.014211         0.011178           -1.271           0.204
    L7.Latency                       -0.000000         0.000000           -0.962           0.336
    L7.Crash Status                   0.047498         0.006857            6.927           0.000
    L7.End of Playback Status        -0.002132         0.002959           -0.721           0.471
    L7.User_ID_N                      0.000002         0.000004            0.437           0.662
    L7.Title_N                       -0.000000         0.000001           -0.357           0.721
    L7.Device_Vendor_N                0.000523         0.000197            2.656           0.008
    L7.Device_Model_N                -0.000016         0.000023           -0.708           0.479
    L7.Content_TV_Show_N             -0.000002         0.000001           -2.317           0.021
    L7.Country_N                     -0.000423         0.000300           -1.410           0.159
    L7.City_N                        -0.000001         0.000006           -0.149           0.881
    L7.Region_N                       0.006953         0.005593            1.243           0.214
    L8.Playtime                       0.000000         0.000000            0.013           0.990
    L8.Interruptions                  0.000001         0.000035            0.034           0.973
    L8.Join Time                      0.000081         0.000160            0.507           0.612
    L8.Buffer Ratio                   0.000131         0.000160            0.818           0.414
    L8.Connection Type                0.000094         0.000139            0.675           0.499
    L8.Device                         0.000181         0.000397            0.457           0.648
    L8.Device Type                    0.000109         0.000323            0.338           0.735
    L8.Browser                       -0.000764         0.000475           -1.609           0.108
    L8.OS                             0.000336         0.001483            0.227           0.821
    L8.OS Version                     0.000033         0.000155            0.211           0.833
    L8.Device ID                      0.000000         0.000001            0.071           0.943
    L8.Happiness Score               -0.000087         0.000132           -0.658           0.511
    L8.Playback Stalls                0.001493         0.002327            0.642           0.521
    L8.Startup Error (Count)          0.023345         0.011170            2.090           0.037
    L8.Latency                       -0.000000         0.000000           -0.834           0.404
    L8.Crash Status                   0.012975         0.006858            1.892           0.058
    L8.End of Playback Status         0.004451         0.002958            1.505           0.132
    L8.User_ID_N                     -0.000002         0.000003           -0.612           0.540
    L8.Title_N                       -0.000002         0.000001           -1.925           0.054
    L8.Device_Vendor_N               -0.000039         0.000196           -0.198           0.843
    L8.Device_Model_N                -0.000017         0.000023           -0.734           0.463
    L8.Content_TV_Show_N             -0.000002         0.000001           -2.168           0.030
    L8.Country_N                     -0.000085         0.000298           -0.285           0.775
    L8.City_N                         0.000010         0.000006            1.705           0.088
    L8.Region_N                      -0.003435         0.005570           -0.617           0.537
    L9.Playtime                       0.000000         0.000000            0.025           0.980
    L9.Interruptions                 -0.000002         0.000035           -0.058           0.954
    L9.Join Time                      0.000052         0.000160            0.325           0.745
    L9.Buffer Ratio                  -0.000295         0.000160           -1.844           0.065
    L9.Connection Type                0.000002         0.000138            0.018           0.986
    L9.Device                        -0.000381         0.000393           -0.970           0.332
    L9.Device Type                   -0.000267         0.000319           -0.837           0.403
    L9.Browser                        0.000931         0.000468            1.991           0.047
    L9.OS                             0.000124         0.001466            0.085           0.932
    L9.OS Version                    -0.000212         0.000154           -1.376           0.169
    L9.Device ID                      0.000001         0.000001            0.872           0.383
    L9.Happiness Score               -0.000084         0.000131           -0.646           0.518
    L9.Playback Stalls                0.003448         0.002319            1.487           0.137
    L9.Startup Error (Count)          0.003624         0.011160            0.325           0.745
    L9.Latency                        0.000000         0.000000            0.259           0.796
    L9.Crash Status                   0.033774         0.006852            4.929           0.000
    L9.End of Playback Status        -0.000608         0.002957           -0.205           0.837
    L9.User_ID_N                     -0.000002         0.000003           -0.617           0.537
    L9.Title_N                        0.000001         0.000001            0.590           0.555
    L9.Device_Vendor_N               -0.000048         0.000193           -0.250           0.802
    L9.Device_Model_N                -0.000006         0.000023           -0.252           0.801
    L9.Content_TV_Show_N             -0.000002         0.000001           -2.687           0.007
    L9.Country_N                     -0.000035         0.000296           -0.118           0.906
    L9.City_N                         0.000006         0.000006            1.151           0.250
    L9.Region_N                       0.006789         0.005534            1.227           0.220
    ============================================================================================
    
    Results for equation End of Playback Status
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             1.777492         0.049116           36.189           0.000
    L1.Playtime                       0.000001         0.000001            1.719           0.086
    L1.Interruptions                 -0.000019         0.000077           -0.249           0.804
    L1.Join Time                      0.000543         0.000351            1.548           0.122
    L1.Buffer Ratio                   0.002208         0.000348            6.343           0.000
    L1.Connection Type               -0.000163         0.000300           -0.541           0.588
    L1.Device                        -0.000492         0.000857           -0.573           0.566
    L1.Device Type                   -0.001291         0.000696           -1.856           0.063
    L1.Browser                       -0.002111         0.001020           -2.069           0.039
    L1.OS                            -0.002653         0.003202           -0.829           0.407
    L1.OS Version                     0.001054         0.000336            3.134           0.002
    L1.Device ID                     -0.000002         0.000002           -0.920           0.358
    L1.Happiness Score                0.000909         0.000285            3.187           0.001
    L1.Playback Stalls                0.005615         0.005062            1.109           0.267
    L1.Startup Error (Count)         -0.143786         0.024383           -5.897           0.000
    L1.Latency                        0.000000         0.000000            0.194           0.846
    L1.Crash Status                   0.176968         0.014926           11.856           0.000
    L1.End of Playback Status         0.076929         0.006431           11.963           0.000
    L1.User_ID_N                      0.000013         0.000007            1.724           0.085
    L1.Title_N                        0.000001         0.000002            0.306           0.760
    L1.Device_Vendor_N                0.001111         0.000421            2.637           0.008
    L1.Device_Model_N                 0.000169         0.000050            3.400           0.001
    L1.Content_TV_Show_N              0.000010         0.000002            5.083           0.000
    L1.Country_N                     -0.002863         0.000646           -4.432           0.000
    L1.City_N                         0.000043         0.000012            3.582           0.000
    L1.Region_N                       0.012939         0.012070            1.072           0.284
    L2.Playtime                      -0.000000         0.000001           -0.223           0.823
    L2.Interruptions                 -0.000011         0.000077           -0.147           0.883
    L2.Join Time                      0.000794         0.000351            2.263           0.024
    L2.Buffer Ratio                   0.000508         0.000349            1.454           0.146
    L2.Connection Type                0.000077         0.000304            0.253           0.800
    L2.Device                         0.001535         0.000866            1.773           0.076
    L2.Device Type                    0.000736         0.000705            1.044           0.297
    L2.Browser                       -0.001491         0.001035           -1.441           0.150
    L2.OS                             0.000263         0.003237            0.081           0.935
    L2.OS Version                     0.000050         0.000339            0.146           0.884
    L2.Device ID                      0.000000         0.000002            0.031           0.976
    L2.Happiness Score                0.000056         0.000289            0.193           0.847
    L2.Playback Stalls                0.005346         0.005062            1.056           0.291
    L2.Startup Error (Count)         -0.099248         0.024397           -4.068           0.000
    L2.Latency                        0.000000         0.000000            1.535           0.125
    L2.Crash Status                   0.037045         0.014964            2.476           0.013
    L2.End of Playback Status         0.058818         0.006445            9.127           0.000
    L2.User_ID_N                      0.000001         0.000008            0.121           0.904
    L2.Title_N                        0.000001         0.000002            0.276           0.782
    L2.Device_Vendor_N               -0.000377         0.000427           -0.884           0.377
    L2.Device_Model_N                -0.000019         0.000050           -0.370           0.711
    L2.Content_TV_Show_N              0.000004         0.000002            1.908           0.056
    L2.Country_N                     -0.002098         0.000651           -3.224           0.001
    L2.City_N                         0.000019         0.000012            1.557           0.119
    L2.Region_N                       0.011987         0.012153            0.986           0.324
    L3.Playtime                       0.000001         0.000001            1.909           0.056
    L3.Interruptions                 -0.000010         0.000077           -0.133           0.894
    L3.Join Time                      0.000067         0.000351            0.191           0.849
    L3.Buffer Ratio                   0.000570         0.000349            1.633           0.102
    L3.Connection Type               -0.000517         0.000307           -1.686           0.092
    L3.Device                        -0.001555         0.000870           -1.787           0.074
    L3.Device Type                    0.001272         0.000710            1.790           0.073
    L3.Browser                        0.001749         0.001041            1.680           0.093
    L3.OS                             0.000215         0.003248            0.066           0.947
    L3.OS Version                    -0.000443         0.000340           -1.303           0.193
    L3.Device ID                     -0.000003         0.000002           -1.388           0.165
    L3.Happiness Score               -0.000414         0.000290           -1.430           0.153
    L3.Playback Stalls                0.001689         0.005063            0.334           0.739
    L3.Startup Error (Count)         -0.071964         0.024398           -2.950           0.003
    L3.Latency                        0.000000         0.000000            1.771           0.077
    L3.Crash Status                   0.014920         0.014959            0.997           0.319
    L3.End of Playback Status         0.034535         0.006451            5.353           0.000
    L3.User_ID_N                     -0.000011         0.000008           -1.385           0.166
    L3.Title_N                        0.000001         0.000002            0.616           0.538
    L3.Device_Vendor_N               -0.000996         0.000430           -2.317           0.020
    L3.Device_Model_N                 0.000070         0.000051            1.387           0.166
    L3.Content_TV_Show_N             -0.000001         0.000002           -0.712           0.477
    L3.Country_N                     -0.000937         0.000654           -1.432           0.152
    L3.City_N                         0.000022         0.000012            1.786           0.074
    L3.Region_N                       0.024252         0.012203            1.987           0.047
    L4.Playtime                       0.000000         0.000001            0.696           0.486
    L4.Interruptions                  0.000009         0.000077            0.113           0.910
    L4.Join Time                     -0.000500         0.000349           -1.432           0.152
    L4.Buffer Ratio                   0.001202         0.000349            3.443           0.001
    L4.Connection Type                0.000377         0.000308            1.226           0.220
    L4.Device                         0.000866         0.000872            0.993           0.321
    L4.Device Type                   -0.000111         0.000713           -0.155           0.876
    L4.Browser                        0.000787         0.001044            0.753           0.451
    L4.OS                            -0.004155         0.003257           -1.276           0.202
    L4.OS Version                     0.000136         0.000340            0.400           0.689
    L4.Device ID                      0.000002         0.000002            0.917           0.359
    L4.Happiness Score               -0.000518         0.000290           -1.786           0.074
    L4.Playback Stalls               -0.010357         0.005061           -2.047           0.041
    L4.Startup Error (Count)         -0.031088         0.024396           -1.274           0.203
    L4.Latency                        0.000000         0.000000            1.599           0.110
    L4.Crash Status                  -0.024507         0.014959           -1.638           0.101
    L4.End of Playback Status         0.048407         0.006453            7.502           0.000
    L4.User_ID_N                      0.000001         0.000008            0.086           0.931
    L4.Title_N                       -0.000000         0.000002           -0.031           0.975
    L4.Device_Vendor_N               -0.000470         0.000431           -1.090           0.276
    L4.Device_Model_N                -0.000073         0.000051           -1.435           0.151
    L4.Content_TV_Show_N              0.000004         0.000002            1.989           0.047
    L4.Country_N                     -0.000425         0.000656           -0.647           0.518
    L4.City_N                        -0.000006         0.000012           -0.503           0.615
    L4.Region_N                       0.007799         0.012237            0.637           0.524
    L5.Playtime                      -0.000000         0.000001           -0.340           0.734
    L5.Interruptions                  0.000010         0.000077            0.125           0.901
    L5.Join Time                     -0.001191         0.000349           -3.410           0.001
    L5.Buffer Ratio                  -0.000411         0.000349           -1.179           0.239
    L5.Connection Type                0.000218         0.000308            0.709           0.478
    L5.Device                        -0.000342         0.000873           -0.392           0.695
    L5.Device Type                    0.000826         0.000714            1.158           0.247
    L5.Browser                       -0.001810         0.001046           -1.731           0.084
    L5.OS                             0.000866         0.003260            0.266           0.790
    L5.OS Version                     0.000208         0.000341            0.609           0.542
    L5.Device ID                     -0.000001         0.000002           -0.408           0.683
    L5.Happiness Score               -0.000190         0.000290           -0.654           0.513
    L5.Playback Stalls               -0.004380         0.004928           -0.889           0.374
    L5.Startup Error (Count)         -0.106748         0.024394           -4.376           0.000
    L5.Latency                       -0.000000         0.000000           -0.057           0.954
    L5.Crash Status                   0.014100         0.014961            0.942           0.346
    L5.End of Playback Status         0.004656         0.006455            0.721           0.471
    L5.User_ID_N                      0.000004         0.000008            0.585           0.559
    L5.Title_N                       -0.000002         0.000002           -0.822           0.411
    L5.Device_Vendor_N                0.000221         0.000431            0.513           0.608
    L5.Device_Model_N                 0.000078         0.000051            1.532           0.126
    L5.Content_TV_Show_N             -0.000001         0.000002           -0.618           0.536
    L5.Country_N                      0.000290         0.000657            0.441           0.659
    L5.City_N                        -0.000004         0.000012           -0.328           0.743
    L5.Region_N                      -0.008890         0.012254           -0.725           0.468
    L6.Playtime                      -0.000000         0.000001           -0.002           0.999
    L6.Interruptions                 -0.000007         0.000077           -0.090           0.929
    L6.Join Time                     -0.000397         0.000349           -1.138           0.255
    L6.Buffer Ratio                   0.000421         0.000349            1.207           0.227
    L6.Connection Type               -0.000013         0.000308           -0.043           0.965
    L6.Device                        -0.000653         0.000872           -0.748           0.454
    L6.Device Type                    0.001820         0.000713            2.551           0.011
    L6.Browser                       -0.002465         0.001044           -2.360           0.018
    L6.OS                             0.007038         0.003257            2.161           0.031
    L6.OS Version                    -0.000418         0.000340           -1.228           0.220
    L6.Device ID                      0.000001         0.000002            0.453           0.650
    L6.Happiness Score               -0.000467         0.000290           -1.612           0.107
    L6.Playback Stalls                0.004032         0.005050            0.798           0.425
    L6.Startup Error (Count)         -0.026468         0.024391           -1.085           0.278
    L6.Latency                       -0.000000         0.000000           -0.386           0.700
    L6.Crash Status                   0.003164         0.014960            0.211           0.833
    L6.End of Playback Status         0.033384         0.006456            5.171           0.000
    L6.User_ID_N                      0.000008         0.000008            1.080           0.280
    L6.Title_N                        0.000000         0.000002            0.189           0.850
    L6.Device_Vendor_N               -0.000579         0.000431           -1.342           0.180
    L6.Device_Model_N                -0.000004         0.000051           -0.071           0.944
    L6.Content_TV_Show_N              0.000001         0.000002            0.446           0.655
    L6.Country_N                     -0.001981         0.000656           -3.020           0.003
    L6.City_N                        -0.000011         0.000012           -0.901           0.367
    L6.Region_N                       0.033878         0.012235            2.769           0.006
    L7.Playtime                       0.000000         0.000001            0.284           0.776
    L7.Interruptions                  0.000005         0.000077            0.069           0.945
    L7.Join Time                      0.000317         0.000349            0.908           0.364
    L7.Buffer Ratio                  -0.000263         0.000349           -0.754           0.451
    L7.Connection Type               -0.000047         0.000307           -0.153           0.878
    L7.Device                         0.000101         0.000870            0.117           0.907
    L7.Device Type                    0.000737         0.000710            1.037           0.300
    L7.Browser                        0.000723         0.001041            0.694           0.487
    L7.OS                            -0.001236         0.003248           -0.381           0.704
    L7.OS Version                     0.000044         0.000340            0.129           0.897
    L7.Device ID                      0.000003         0.000002            1.301           0.193
    L7.Happiness Score                0.000576         0.000290            1.991           0.046
    L7.Playback Stalls                0.002388         0.005076            0.470           0.638
    L7.Startup Error (Count)         -0.036951         0.024389           -1.515           0.130
    L7.Latency                       -0.000000         0.000000           -0.197           0.844
    L7.Crash Status                  -0.005383         0.014960           -0.360           0.719
    L7.End of Playback Status         0.016597         0.006455            2.571           0.010
    L7.User_ID_N                      0.000001         0.000008            0.079           0.937
    L7.Title_N                       -0.000001         0.000002           -0.303           0.762
    L7.Device_Vendor_N               -0.000269         0.000430           -0.626           0.532
    L7.Device_Model_N                 0.000015         0.000051            0.306           0.760
    L7.Content_TV_Show_N              0.000001         0.000002            0.437           0.662
    L7.Country_N                      0.000771         0.000654            1.179           0.239
    L7.City_N                        -0.000013         0.000012           -1.058           0.290
    L7.Region_N                      -0.007440         0.012202           -0.610           0.542
    L8.Playtime                       0.000001         0.000001            1.062           0.288
    L8.Interruptions                 -0.000004         0.000077           -0.056           0.956
    L8.Join Time                      0.000409         0.000349            1.172           0.241
    L8.Buffer Ratio                  -0.000131         0.000349           -0.375           0.708
    L8.Connection Type                0.000136         0.000304            0.448           0.654
    L8.Device                         0.000700         0.000866            0.809           0.419
    L8.Device Type                   -0.000411         0.000705           -0.583           0.560
    L8.Browser                       -0.000514         0.001036           -0.496           0.620
    L8.OS                             0.001524         0.003236            0.471           0.638
    L8.OS Version                    -0.000033         0.000339           -0.097           0.922
    L8.Device ID                     -0.000000         0.000002           -0.126           0.900
    L8.Happiness Score               -0.000023         0.000289           -0.080           0.937
    L8.Playback Stalls                0.000434         0.005076            0.085           0.932
    L8.Startup Error (Count)         -0.077009         0.024371           -3.160           0.002
    L8.Latency                        0.000000         0.000000            1.526           0.127
    L8.Crash Status                   0.006846         0.014963            0.458           0.647
    L8.End of Playback Status        -0.000264         0.006454           -0.041           0.967
    L8.User_ID_N                      0.000006         0.000008            0.779           0.436
    L8.Title_N                       -0.000001         0.000002           -0.482           0.630
    L8.Device_Vendor_N                0.000010         0.000427            0.023           0.981
    L8.Device_Model_N                -0.000048         0.000050           -0.954           0.340
    L8.Content_TV_Show_N              0.000009         0.000002            4.303           0.000
    L8.Country_N                     -0.001034         0.000651           -1.590           0.112
    L8.City_N                        -0.000010         0.000012           -0.811           0.418
    L8.Region_N                       0.012746         0.012152            1.049           0.294
    L9.Playtime                      -0.000000         0.000001           -0.708           0.479
    L9.Interruptions                  0.000022         0.000077            0.290           0.772
    L9.Join Time                      0.000315         0.000349            0.902           0.367
    L9.Buffer Ratio                   0.000139         0.000348            0.399           0.690
    L9.Connection Type               -0.000247         0.000300           -0.821           0.412
    L9.Device                         0.000069         0.000857            0.081           0.936
    L9.Device Type                    0.001245         0.000695            1.791           0.073
    L9.Browser                       -0.001090         0.001021           -1.068           0.285
    L9.OS                             0.006356         0.003198            1.987           0.047
    L9.OS Version                    -0.000520         0.000336           -1.546           0.122
    L9.Device ID                     -0.000001         0.000002           -0.256           0.798
    L9.Happiness Score                0.000213         0.000285            0.749           0.454
    L9.Playback Stalls                0.005236         0.005059            1.035           0.301
    L9.Startup Error (Count)         -0.028298         0.024349           -1.162           0.245
    L9.Latency                        0.000000         0.000000            0.421           0.674
    L9.Crash Status                   0.004530         0.014950            0.303           0.762
    L9.End of Playback Status         0.021768         0.006451            3.374           0.001
    L9.User_ID_N                     -0.000005         0.000007           -0.646           0.518
    L9.Title_N                        0.000001         0.000002            0.428           0.669
    L9.Device_Vendor_N                0.000290         0.000421            0.687           0.492
    L9.Device_Model_N                -0.000064         0.000050           -1.286           0.199
    L9.Content_TV_Show_N              0.000001         0.000002            0.701           0.483
    L9.Country_N                     -0.001679         0.000646           -2.600           0.009
    L9.City_N                        -0.000002         0.000012           -0.168           0.867
    L9.Region_N                       0.033434         0.012074            2.769           0.006
    ============================================================================================
    
    Results for equation User_ID_N
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                           102.711986        24.455076            4.200           0.000
    L1.Playtime                       0.000592         0.000298            1.985           0.047
    L1.Interruptions                 -0.012941         0.038265           -0.338           0.735
    L1.Join Time                      0.469072         0.174708            2.685           0.007
    L1.Buffer Ratio                   0.221339         0.173324            1.277           0.202
    L1.Connection Type                0.458341         0.149517            3.065           0.002
    L1.Device                        -0.273844         0.426942           -0.641           0.521
    L1.Device Type                   -0.104404         0.346354           -0.301           0.763
    L1.Browser                        1.089096         0.508084            2.144           0.032
    L1.OS                             2.119489         1.594210            1.329           0.184
    L1.OS Version                    -0.117408         0.167475           -0.701           0.483
    L1.Device ID                     -0.001900         0.001171           -1.622           0.105
    L1.Happiness Score                0.104256         0.141991            0.734           0.463
    L1.Playback Stalls               -4.847469         2.520441           -1.923           0.054
    L1.Startup Error (Count)         -0.965602        12.140442           -0.080           0.937
    L1.Latency                        0.000021         0.000022            0.943           0.346
    L1.Crash Status                  -9.345154         7.431921           -1.257           0.209
    L1.End of Playback Status         0.982611         3.201814            0.307           0.759
    L1.User_ID_N                      0.177066         0.003724           47.547           0.000
    L1.Title_N                        0.006342         0.001010            6.279           0.000
    L1.Device_Vendor_N                0.117176         0.209717            0.559           0.576
    L1.Device_Model_N                -0.052291         0.024727           -2.115           0.034
    L1.Content_TV_Show_N             -0.000548         0.001001           -0.548           0.584
    L1.Country_N                     -0.610445         0.321682           -1.898           0.058
    L1.City_N                        -0.020089         0.005970           -3.365           0.001
    L1.Region_N                       3.956285         6.009914            0.658           0.510
    L2.Playtime                      -0.000636         0.000298           -2.134           0.033
    L2.Interruptions                  0.021798         0.038256            0.570           0.569
    L2.Join Time                     -0.155452         0.174758           -0.890           0.374
    L2.Buffer Ratio                  -0.193933         0.173736           -1.116           0.264
    L2.Connection Type                0.418301         0.151412            2.763           0.006
    L2.Device                         0.070350         0.431094            0.163           0.870
    L2.Device Type                   -0.061730         0.351051           -0.176           0.860
    L2.Browser                       -0.638574         0.515368           -1.239           0.215
    L2.OS                             2.025520         1.611600            1.257           0.209
    L2.OS Version                    -0.008837         0.168636           -0.052           0.958
    L2.Device ID                     -0.000635         0.001185           -0.535           0.592
    L2.Happiness Score                0.109272         0.143862            0.760           0.448
    L2.Playback Stalls               -0.208888         2.520614           -0.083           0.934
    L2.Startup Error (Count)         -0.213320        12.147416           -0.018           0.986
    L2.Latency                        0.000042         0.000022            1.893           0.058
    L2.Crash Status                  -0.395708         7.450400           -0.053           0.958
    L2.End of Playback Status        -1.036964         3.208849           -0.323           0.747
    L2.User_ID_N                      0.131262         0.003782           34.711           0.000
    L2.Title_N                        0.004032         0.001011            3.986           0.000
    L2.Device_Vendor_N                0.183936         0.212442            0.866           0.387
    L2.Device_Model_N                -0.021990         0.025022           -0.879           0.379
    L2.Content_TV_Show_N             -0.001350         0.001005           -1.343           0.179
    L2.Country_N                      0.966380         0.323990            2.983           0.003
    L2.City_N                        -0.003540         0.006062           -0.584           0.559
    L2.Region_N                      -8.395503         6.050926           -1.387           0.165
    L3.Playtime                       0.000363         0.000298            1.217           0.224
    L3.Interruptions                  0.001682         0.038256            0.044           0.965
    L3.Join Time                      0.149629         0.174752            0.856           0.392
    L3.Buffer Ratio                  -0.252158         0.173805           -1.451           0.147
    L3.Connection Type                0.091324         0.152642            0.598           0.550
    L3.Device                        -0.778449         0.433298           -1.797           0.072
    L3.Device Type                   -0.347479         0.353744           -0.982           0.326
    L3.Browser                       -0.534569         0.518263           -1.031           0.302
    L3.OS                             2.434271         1.617359            1.505           0.132
    L3.OS Version                    -0.160837         0.169078           -0.951           0.341
    L3.Device ID                     -0.001183         0.001194           -0.991           0.322
    L3.Happiness Score                0.219422         0.144300            1.521           0.128
    L3.Playback Stalls                3.486067         2.520753            1.383           0.167
    L3.Startup Error (Count)          0.110585        12.147623            0.009           0.993
    L3.Latency                        0.000009         0.000022            0.405           0.685
    L3.Crash Status                   0.341597         7.448220            0.046           0.963
    L3.End of Playback Status         2.323193         3.212171            0.723           0.470
    L3.User_ID_N                      0.094408         0.003809           24.784           0.000
    L3.Title_N                        0.002472         0.001012            2.443           0.015
    L3.Device_Vendor_N               -0.412335         0.213937           -1.927           0.054
    L3.Device_Model_N                 0.006062         0.025153            0.241           0.810
    L3.Content_TV_Show_N             -0.000315         0.001008           -0.313           0.754
    L3.Country_N                      0.687798         0.325849            2.111           0.035
    L3.City_N                         0.009895         0.006108            1.620           0.105
    L3.Region_N                     -13.309330         6.075668           -2.191           0.028
    L4.Playtime                       0.000170         0.000298            0.569           0.570
    L4.Interruptions                 -0.000783         0.038256           -0.020           0.984
    L4.Join Time                      0.263521         0.173821            1.516           0.130
    L4.Buffer Ratio                   0.116366         0.173793            0.670           0.503
    L4.Connection Type               -0.148740         0.153247           -0.971           0.332
    L4.Device                        -0.529586         0.434301           -1.219           0.223
    L4.Device Type                   -0.012187         0.355002           -0.034           0.973
    L4.Browser                       -0.322984         0.520036           -0.621           0.535
    L4.OS                             0.598574         1.621763            0.369           0.712
    L4.OS Version                    -0.087438         0.169514           -0.516           0.606
    L4.Device ID                     -0.000609         0.001198           -0.508           0.611
    L4.Happiness Score               -0.134306         0.144445           -0.930           0.352
    L4.Playback Stalls               -6.117558         2.519741           -2.428           0.015
    L4.Startup Error (Count)         -8.633779        12.147023           -0.711           0.477
    L4.Latency                        0.000000         0.000022            0.009           0.993
    L4.Crash Status                   2.626423         7.448186            0.353           0.724
    L4.End of Playback Status        -3.411155         3.212861           -1.062           0.288
    L4.User_ID_N                      0.067167         0.003821           17.576           0.000
    L4.Title_N                       -0.000005         0.001012           -0.005           0.996
    L4.Device_Vendor_N               -0.290956         0.214675           -1.355           0.175
    L4.Device_Model_N                -0.011258         0.025216           -0.446           0.655
    L4.Content_TV_Show_N             -0.001091         0.001009           -1.082           0.279
    L4.Country_N                      0.544120         0.326673            1.666           0.096
    L4.City_N                         0.000443         0.006127            0.072           0.942
    L4.Region_N                       5.884613         6.092876            0.966           0.334
    L5.Playtime                       0.000352         0.000298            1.179           0.238
    L5.Interruptions                  0.002539         0.038257            0.066           0.947
    L5.Join Time                     -0.294758         0.173888           -1.695           0.090
    L5.Buffer Ratio                  -0.067441         0.173541           -0.389           0.698
    L5.Connection Type               -0.105724         0.153376           -0.689           0.491
    L5.Device                         0.431491         0.434606            0.993           0.321
    L5.Device Type                    0.030829         0.355436            0.087           0.931
    L5.Browser                        0.655798         0.520845            1.259           0.208
    L5.OS                            -0.830023         1.623381           -0.511           0.609
    L5.OS Version                    -0.007746         0.169686           -0.046           0.964
    L5.Device ID                     -0.000631         0.001199           -0.526           0.599
    L5.Happiness Score                0.095639         0.144463            0.662           0.508
    L5.Playback Stalls                0.927366         2.453614            0.378           0.705
    L5.Startup Error (Count)         -9.975413        12.145882           -0.821           0.411
    L5.Latency                        0.000019         0.000022            0.865           0.387
    L5.Crash Status                   5.423313         7.449195            0.728           0.467
    L5.End of Playback Status        -3.682755         3.214125           -1.146           0.252
    L5.User_ID_N                      0.049302         0.003825           12.890           0.000
    L5.Title_N                        0.000449         0.001012            0.444           0.657
    L5.Device_Vendor_N                0.109483         0.214828            0.510           0.610
    L5.Device_Model_N                 0.016397         0.025232            0.650           0.516
    L5.Content_TV_Show_N              0.002614         0.001009            2.590           0.010
    L5.Country_N                      0.040011         0.327281            0.122           0.903
    L5.City_N                         0.009190         0.006135            1.498           0.134
    L5.Region_N                      -6.757527         6.101143           -1.108           0.268
    L6.Playtime                      -0.000504         0.000298           -1.690           0.091
    L6.Interruptions                  0.014267         0.038257            0.373           0.709
    L6.Join Time                     -0.188707         0.173957           -1.085           0.278
    L6.Buffer Ratio                  -0.154405         0.173548           -0.890           0.374
    L6.Connection Type                0.045924         0.153249            0.300           0.764
    L6.Device                         0.272749         0.434341            0.628           0.530
    L6.Device Type                   -0.219599         0.355102           -0.618           0.536
    L6.Browser                       -0.725324         0.520010           -1.395           0.163
    L6.OS                            -0.103489         1.621772           -0.064           0.949
    L6.OS Version                     0.059944         0.169528            0.354           0.724
    L6.Device ID                     -0.001014         0.001198           -0.846           0.397
    L6.Happiness Score                0.242957         0.144359            1.683           0.092
    L6.Playback Stalls                1.370603         2.514362            0.545           0.586
    L6.Startup Error (Count)         -0.341766        12.144183           -0.028           0.978
    L6.Latency                       -0.000042         0.000022           -1.924           0.054
    L6.Crash Status                   0.789965         7.448658            0.106           0.916
    L6.End of Playback Status        -0.547629         3.214591           -0.170           0.865
    L6.User_ID_N                      0.045428         0.003822           11.887           0.000
    L6.Title_N                       -0.000622         0.001012           -0.614           0.539
    L6.Device_Vendor_N                0.154557         0.214651            0.720           0.472
    L6.Device_Model_N                -0.015906         0.025211           -0.631           0.528
    L6.Content_TV_Show_N              0.000458         0.001009            0.454           0.650
    L6.Country_N                      1.090999         0.326619            3.340           0.001
    L6.City_N                         0.008993         0.006127            1.468           0.142
    L6.Region_N                     -19.219420         6.092061           -3.155           0.002
    L7.Playtime                       0.000132         0.000298            0.443           0.657
    L7.Interruptions                  0.002823         0.038258            0.074           0.941
    L7.Join Time                      0.146969         0.173936            0.845           0.398
    L7.Buffer Ratio                   0.311401         0.173577            1.794           0.073
    L7.Connection Type               -0.086324         0.152639           -0.566           0.572
    L7.Device                         0.968035         0.433168            2.235           0.025
    L7.Device Type                   -0.201718         0.353754           -0.570           0.569
    L7.Browser                        0.044102         0.518344            0.085           0.932
    L7.OS                            -3.892509         1.617243           -2.407           0.016
    L7.OS Version                     0.308495         0.169108            1.824           0.068
    L7.Device ID                      0.000428         0.001194            0.359           0.720
    L7.Happiness Score                0.227242         0.144164            1.576           0.115
    L7.Playback Stalls                1.430717         2.527433            0.566           0.571
    L7.Startup Error (Count)          7.617255        12.143153            0.627           0.530
    L7.Latency                        0.000018         0.000022            0.815           0.415
    L7.Crash Status                  -3.601973         7.448523           -0.484           0.629
    L7.End of Playback Status         1.308197         3.214149            0.407           0.684
    L7.User_ID_N                      0.037752         0.003809            9.911           0.000
    L7.Title_N                        0.000285         0.001012            0.282           0.778
    L7.Device_Vendor_N                0.129285         0.213965            0.604           0.546
    L7.Device_Model_N                 0.022595         0.025148            0.898           0.369
    L7.Content_TV_Show_N             -0.000510         0.001008           -0.506           0.613
    L7.Country_N                      0.400154         0.325741            1.228           0.219
    L7.City_N                        -0.007350         0.006108           -1.203           0.229
    L7.Region_N                      -0.911141         6.075329           -0.150           0.881
    L8.Playtime                      -0.000062         0.000298           -0.208           0.836
    L8.Interruptions                 -0.018710         0.038257           -0.489           0.625
    L8.Join Time                     -0.147407         0.173919           -0.848           0.397
    L8.Buffer Ratio                   0.026148         0.173549            0.151           0.880
    L8.Connection Type               -0.122847         0.151426           -0.811           0.417
    L8.Device                        -0.407165         0.430958           -0.945           0.345
    L8.Device Type                   -0.265721         0.351091           -0.757           0.449
    L8.Browser                        0.218275         0.515591            0.423           0.672
    L8.OS                             0.522365         1.611272            0.324           0.746
    L8.OS Version                    -0.075218         0.168609           -0.446           0.656
    L8.Device ID                     -0.001152         0.001185           -0.972           0.331
    L8.Happiness Score                0.259398         0.143776            1.804           0.071
    L8.Playback Stalls               -1.206206         2.527487           -0.477           0.633
    L8.Startup Error (Count)          6.268711        12.134515            0.517           0.605
    L8.Latency                       -0.000029         0.000022           -1.341           0.180
    L8.Crash Status                   0.166589         7.449932            0.022           0.982
    L8.End of Playback Status         0.662116         3.213360            0.206           0.837
    L8.User_ID_N                      0.032989         0.003782            8.723           0.000
    L8.Title_N                        0.000546         0.001011            0.540           0.589
    L8.Device_Vendor_N                0.203475         0.212457            0.958           0.338
    L8.Device_Model_N                 0.049322         0.025018            1.971           0.049
    L8.Content_TV_Show_N              0.000291         0.001005            0.290           0.772
    L8.Country_N                      0.083970         0.323892            0.259           0.795
    L8.City_N                        -0.010244         0.006061           -1.690           0.091
    L8.Region_N                       4.871742         6.050279            0.805           0.421
    L9.Playtime                       0.000158         0.000298            0.531           0.596
    L9.Interruptions                 -0.001706         0.038257           -0.045           0.964
    L9.Join Time                     -0.078804         0.173821           -0.453           0.650
    L9.Buffer Ratio                   0.045974         0.173514            0.265           0.791
    L9.Connection Type                0.391433         0.149532            2.618           0.009
    L9.Device                         1.386869         0.426669            3.250           0.001
    L9.Device Type                   -0.176707         0.346164           -0.510           0.610
    L9.Browser                       -0.540885         0.508168           -1.064           0.287
    L9.OS                            -1.714040         1.592467           -1.076           0.282
    L9.OS Version                     0.132716         0.167418            0.793           0.428
    L9.Device ID                      0.002504         0.001172            2.136           0.033
    L9.Happiness Score                0.086807         0.141825            0.612           0.540
    L9.Playback Stalls                1.759458         2.518726            0.699           0.485
    L9.Startup Error (Count)         20.896687        12.123579            1.724           0.085
    L9.Latency                       -0.000002         0.000022           -0.092           0.926
    L9.Crash Status                  -7.843173         7.443725           -1.054           0.292
    L9.End of Playback Status         5.900217         3.212204            1.837           0.066
    L9.User_ID_N                      0.034725         0.003726            9.319           0.000
    L9.Title_N                        0.001148         0.001010            1.137           0.256
    L9.Device_Vendor_N                0.274394         0.209718            1.308           0.191
    L9.Device_Model_N                -0.021098         0.024726           -0.853           0.394
    L9.Content_TV_Show_N             -0.000320         0.001001           -0.319           0.749
    L9.Country_N                      0.276601         0.321632            0.860           0.390
    L9.City_N                        -0.010515         0.005975           -1.760           0.078
    L9.Region_N                     -11.397055         6.011535           -1.896           0.058
    ============================================================================================
    
    Results for equation Title_N
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                           665.880829        89.100553            7.473           0.000
    L1.Playtime                       0.001504         0.001086            1.385           0.166
    L1.Interruptions                 -0.033350         0.139415           -0.239           0.811
    L1.Join Time                      0.344632         0.636536            0.541           0.588
    L1.Buffer Ratio                  -0.636546         0.631494           -1.008           0.313
    L1.Connection Type               -0.914260         0.544758           -1.678           0.093
    L1.Device                         3.312712         1.555537            2.130           0.033
    L1.Device Type                    1.767811         1.261918            1.401           0.161
    L1.Browser                       -2.361156         1.851172           -1.275           0.202
    L1.OS                            -0.330890         5.808404           -0.057           0.955
    L1.OS Version                     0.521019         0.610185            0.854           0.393
    L1.Device ID                      0.012796         0.004267            2.999           0.003
    L1.Happiness Score               -7.475901         0.517335          -14.451           0.000
    L1.Playback Stalls               -0.308368         9.183071           -0.034           0.973
    L1.Startup Error (Count)        -51.155748        44.232949           -1.157           0.247
    L1.Latency                        0.000111         0.000080            1.392           0.164
    L1.Crash Status                   0.263934        27.077742            0.010           0.992
    L1.End of Playback Status       -16.916010        11.665611           -1.450           0.147
    L1.User_ID_N                      0.026832         0.013568            1.978           0.048
    L1.Title_N                        0.034936         0.003680            9.494           0.000
    L1.Device_Vendor_N                3.638726         0.764091            4.762           0.000
    L1.Device_Model_N                 0.029434         0.090091            0.327           0.744
    L1.Content_TV_Show_N             -0.001282         0.003647           -0.351           0.725
    L1.Country_N                     -2.334948         1.172029           -1.992           0.046
    L1.City_N                         0.059575         0.021753            2.739           0.006
    L1.Region_N                     -41.126708        21.896748           -1.878           0.060
    L2.Playtime                      -0.000856         0.001087           -0.788           0.431
    L2.Interruptions                 -0.114366         0.139385           -0.821           0.412
    L2.Join Time                     -0.464992         0.636718           -0.730           0.465
    L2.Buffer Ratio                   1.274702         0.632998            2.014           0.044
    L2.Connection Type               -1.364347         0.551658           -2.473           0.013
    L2.Device                        -2.050217         1.570664           -1.305           0.192
    L2.Device Type                   -0.025766         1.279033           -0.020           0.984
    L2.Browser                        2.740904         1.877710            1.460           0.144
    L2.OS                            -2.247230         5.871765           -0.383           0.702
    L2.OS Version                    -0.028345         0.614416           -0.046           0.963
    L2.Device ID                      0.003903         0.004318            0.904           0.366
    L2.Happiness Score               -5.276568         0.524153          -10.067           0.000
    L2.Playback Stalls               -8.051473         9.183700           -0.877           0.381
    L2.Startup Error (Count)         25.847522        44.258357            0.584           0.559
    L2.Latency                        0.000239         0.000080            2.990           0.003
    L2.Crash Status                 -14.829506        27.145072           -0.546           0.585
    L2.End of Playback Status        16.880968        11.691241            1.444           0.149
    L2.User_ID_N                      0.012235         0.013778            0.888           0.375
    L2.Title_N                        0.037316         0.003685           10.126           0.000
    L2.Device_Vendor_N                0.321872         0.774019            0.416           0.678
    L2.Device_Model_N                 0.170075         0.091166            1.866           0.062
    L2.Content_TV_Show_N              0.004426         0.003662            1.209           0.227
    L2.Country_N                     -0.527234         1.180438           -0.447           0.655
    L2.City_N                        -0.000297         0.022085           -0.013           0.989
    L2.Region_N                      -1.974679        22.046174           -0.090           0.929
    L3.Playtime                      -0.000513         0.001087           -0.472           0.637
    L3.Interruptions                 -0.127347         0.139382           -0.914           0.361
    L3.Join Time                      1.922734         0.636697            3.020           0.003
    L3.Buffer Ratio                  -0.981770         0.633246           -1.550           0.121
    L3.Connection Type               -0.277368         0.556143           -0.499           0.618
    L3.Device                         0.270561         1.578694            0.171           0.864
    L3.Device Type                   -1.346461         1.288843           -1.045           0.296
    L3.Browser                       -1.807651         1.888261           -0.957           0.338
    L3.OS                            -2.414794         5.892746           -0.410           0.682
    L3.OS Version                     0.593161         0.616027            0.963           0.336
    L3.Device ID                     -0.011312         0.004351           -2.600           0.009
    L3.Happiness Score               -3.031023         0.525749           -5.765           0.000
    L3.Playback Stalls                4.266596         9.184208            0.465           0.642
    L3.Startup Error (Count)        -32.073499        44.259111           -0.725           0.469
    L3.Latency                        0.000226         0.000080            2.829           0.005
    L3.Crash Status                   6.682019        27.137128            0.246           0.806
    L3.End of Playback Status         8.798237        11.703345            0.752           0.452
    L3.User_ID_N                     -0.000271         0.013879           -0.019           0.984
    L3.Title_N                        0.016299         0.003688            4.420           0.000
    L3.Device_Vendor_N                0.399832         0.779466            0.513           0.608
    L3.Device_Model_N                -0.012969         0.091643           -0.142           0.887
    L3.Content_TV_Show_N              0.005369         0.003671            1.463           0.144
    L3.Country_N                      0.873262         1.187210            0.736           0.462
    L3.City_N                         0.031449         0.022253            1.413           0.158
    L3.Region_N                     -28.164386        22.136321           -1.272           0.203
    L4.Playtime                      -0.000242         0.001087           -0.223           0.823
    L4.Interruptions                  0.186791         0.139382            1.340           0.180
    L4.Join Time                     -0.061244         0.633307           -0.097           0.923
    L4.Buffer Ratio                   0.455301         0.633202            0.719           0.472
    L4.Connection Type               -0.521481         0.558347           -0.934           0.350
    L4.Device                        -0.983573         1.582349           -0.622           0.534
    L4.Device Type                   -3.635800         1.293428           -2.811           0.005
    L4.Browser                       -0.039962         1.894719           -0.021           0.983
    L4.OS                            -3.306885         5.908793           -0.560           0.576
    L4.OS Version                     0.779055         0.617615            1.261           0.207
    L4.Device ID                     -0.009184         0.004365           -2.104           0.035
    L4.Happiness Score               -0.722142         0.526275           -1.372           0.170
    L4.Playback Stalls                5.923862         9.180522            0.645           0.519
    L4.Startup Error (Count)         60.575746        44.256926            1.369           0.171
    L4.Latency                       -0.000073         0.000080           -0.911           0.363
    L4.Crash Status                 -10.030679        27.137004           -0.370           0.712
    L4.End of Playback Status        24.497117        11.705860            2.093           0.036
    L4.User_ID_N                     -0.008215         0.013923           -0.590           0.555
    L4.Title_N                        0.022229         0.003686            6.030           0.000
    L4.Device_Vendor_N                0.546068         0.782154            0.698           0.485
    L4.Device_Model_N                 0.041485         0.091873            0.452           0.652
    L4.Content_TV_Show_N              0.003935         0.003674            1.071           0.284
    L4.Country_N                      0.503862         1.190213            0.423           0.672
    L4.City_N                         0.010608         0.022323            0.475           0.635
    L4.Region_N                      -1.722099        22.199016           -0.078           0.938
    L5.Playtime                       0.000340         0.001087            0.313           0.754
    L5.Interruptions                 -0.193724         0.139386           -1.390           0.165
    L5.Join Time                      0.542772         0.633549            0.857           0.392
    L5.Buffer Ratio                  -0.645180         0.632284           -1.020           0.308
    L5.Connection Type                0.020889         0.558816            0.037           0.970
    L5.Device                         2.240489         1.583459            1.415           0.157
    L5.Device Type                   -2.320643         1.295010           -1.792           0.073
    L5.Browser                       -2.929298         1.897665           -1.544           0.123
    L5.OS                            -7.578308         5.914689           -1.281           0.200
    L5.OS Version                     1.302724         0.618241            2.107           0.035
    L5.Device ID                      0.000228         0.004368            0.052           0.958
    L5.Happiness Score                0.300626         0.526343            0.571           0.568
    L5.Playback Stalls               11.767897         8.939592            1.316           0.188
    L5.Startup Error (Count)       -100.330154        44.252768           -2.267           0.023
    L5.Latency                        0.000070         0.000080            0.879           0.379
    L5.Crash Status                  32.390024        27.140680            1.193           0.233
    L5.End of Playback Status       -26.003560        11.710465           -2.221           0.026
    L5.User_ID_N                      0.014969         0.013936            1.074           0.283
    L5.Title_N                        0.022034         0.003686            5.977           0.000
    L5.Device_Vendor_N                1.546293         0.782714            1.976           0.048
    L5.Device_Model_N                 0.072196         0.091931            0.785           0.432
    L5.Content_TV_Show_N             -0.000692         0.003677           -0.188           0.851
    L5.Country_N                      0.659681         1.192427            0.553           0.580
    L5.City_N                         0.027502         0.022352            1.230           0.219
    L5.Region_N                     -43.415729        22.229135           -1.953           0.051
    L6.Playtime                      -0.000731         0.001087           -0.673           0.501
    L6.Interruptions                  0.251359         0.139386            1.803           0.071
    L6.Join Time                      0.863607         0.633802            1.363           0.173
    L6.Buffer Ratio                  -0.346434         0.632310           -0.548           0.584
    L6.Connection Type               -1.643321         0.558352           -2.943           0.003
    L6.Device                         2.583800         1.582496            1.633           0.103
    L6.Device Type                   -1.956748         1.293794           -1.512           0.130
    L6.Browser                        0.754276         1.894624            0.398           0.691
    L6.OS                            -3.537069         5.908826           -0.599           0.549
    L6.OS Version                    -0.265331         0.617664           -0.430           0.668
    L6.Device ID                      0.006890         0.004365            1.579           0.114
    L6.Happiness Score                0.842126         0.525962            1.601           0.109
    L6.Playback Stalls               13.442140         9.160922            1.467           0.142
    L6.Startup Error (Count)        -44.949396        44.246578           -1.016           0.310
    L6.Latency                       -0.000289         0.000080           -3.621           0.000
    L6.Crash Status                  15.281838        27.138724            0.563           0.573
    L6.End of Playback Status       -15.442117        11.712163           -1.318           0.187
    L6.User_ID_N                     -0.000963         0.013923           -0.069           0.945
    L6.Title_N                        0.034239         0.003687            9.285           0.000
    L6.Device_Vendor_N               -0.400418         0.782069           -0.512           0.609
    L6.Device_Model_N                -0.139076         0.091856           -1.514           0.130
    L6.Content_TV_Show_N             -0.002820         0.003675           -0.767           0.443
    L6.Country_N                      3.038639         1.190016            2.553           0.011
    L6.City_N                        -0.020751         0.022324           -0.930           0.353
    L6.Region_N                     -24.095946        22.196046           -1.086           0.278
    L7.Playtime                       0.001228         0.001087            1.130           0.259
    L7.Interruptions                 -0.189795         0.139392           -1.362           0.173
    L7.Join Time                     -0.853948         0.633726           -1.348           0.178
    L7.Buffer Ratio                   0.770877         0.632416            1.219           0.223
    L7.Connection Type                0.483808         0.556131            0.870           0.384
    L7.Device                         1.125634         1.578221            0.713           0.476
    L7.Device Type                   -0.503171         1.288882           -0.390           0.696
    L7.Browser                        0.167509         1.888555            0.089           0.929
    L7.OS                             0.108840         5.892324            0.018           0.985
    L7.OS Version                     0.104850         0.616134            0.170           0.865
    L7.Device ID                      0.010072         0.004350            2.315           0.021
    L7.Happiness Score                0.205972         0.525253            0.392           0.695
    L7.Playback Stalls              -11.721498         9.208547           -1.273           0.203
    L7.Startup Error (Count)          8.163378        44.242824            0.185           0.854
    L7.Latency                       -0.000086         0.000080           -1.080           0.280
    L7.Crash Status                 -16.110487        27.138231           -0.594           0.553
    L7.End of Playback Status        -4.335409        11.710552           -0.370           0.711
    L7.User_ID_N                      0.017609         0.013878            1.269           0.205
    L7.Title_N                        0.041219         0.003688           11.177           0.000
    L7.Device_Vendor_N                0.592992         0.779568            0.761           0.447
    L7.Device_Model_N                -0.047765         0.091627           -0.521           0.602
    L7.Content_TV_Show_N              0.001047         0.003671            0.285           0.775
    L7.Country_N                      1.008230         1.186818            0.850           0.396
    L7.City_N                         0.045771         0.022253            2.057           0.040
    L7.Region_N                     -44.911507        22.135085           -2.029           0.042
    L8.Playtime                       0.000183         0.001087            0.169           0.866
    L8.Interruptions                  0.170404         0.139388            1.223           0.222
    L8.Join Time                      1.291994         0.633662            2.039           0.041
    L8.Buffer Ratio                  -0.834148         0.632317           -1.319           0.187
    L8.Connection Type               -0.505016         0.551711           -0.915           0.360
    L8.Device                         0.337175         1.570169            0.215           0.830
    L8.Device Type                   -1.875146         1.279178           -1.466           0.143
    L8.Browser                       -2.305561         1.878524           -1.227           0.220
    L8.OS                            -1.794590         5.870570           -0.306           0.760
    L8.OS Version                    -0.090906         0.614318           -0.148           0.882
    L8.Device ID                      0.007348         0.004317            1.702           0.089
    L8.Happiness Score                0.272652         0.523839            0.520           0.603
    L8.Playback Stalls               16.528649         9.208743            1.795           0.073
    L8.Startup Error (Count)         18.590223        44.211352            0.420           0.674
    L8.Latency                       -0.000199         0.000080           -2.495           0.013
    L8.Crash Status                 -22.479831        27.143365           -0.828           0.408
    L8.End of Playback Status         1.786905        11.707678            0.153           0.879
    L8.User_ID_N                     -0.044201         0.013778           -3.208           0.001
    L8.Title_N                        0.029187         0.003685            7.921           0.000
    L8.Device_Vendor_N               -1.438899         0.774074           -1.859           0.063
    L8.Device_Model_N                 0.026933         0.091152            0.295           0.768
    L8.Content_TV_Show_N              0.003503         0.003661            0.957           0.339
    L8.Country_N                      3.071967         1.180079            2.603           0.009
    L8.City_N                         0.039389         0.022082            1.784           0.074
    L8.Region_N                      -5.998657        22.043817           -0.272           0.786
    L9.Playtime                       0.000972         0.001086            0.895           0.371
    L9.Interruptions                 -0.173460         0.139389           -1.244           0.213
    L9.Join Time                     -0.122316         0.633307           -0.193           0.847
    L9.Buffer Ratio                  -0.683604         0.632187           -1.081           0.280
    L9.Connection Type               -0.938593         0.544810           -1.723           0.085
    L9.Device                         0.508626         1.554543            0.327           0.744
    L9.Device Type                   -0.994260         1.261227           -0.788           0.431
    L9.Browser                        0.296688         1.851478            0.160           0.873
    L9.OS                            -7.524590         5.802056           -1.297           0.195
    L9.OS Version                     1.006376         0.609976            1.650           0.099
    L9.Device ID                     -0.000001         0.004270           -0.000           1.000
    L9.Happiness Score               -1.859808         0.516730           -3.599           0.000
    L9.Playback Stalls              -23.264805         9.176822           -2.535           0.011
    L9.Startup Error (Count)        -13.177063        44.171508           -0.298           0.765
    L9.Latency                        0.000193         0.000080            2.412           0.016
    L9.Crash Status                  12.948443        27.120749            0.477           0.633
    L9.End of Playback Status        -2.679637        11.703467           -0.229           0.819
    L9.User_ID_N                     -0.007742         0.013576           -0.570           0.568
    L9.Title_N                        0.021103         0.003678            5.737           0.000
    L9.Device_Vendor_N                0.749017         0.764096            0.980           0.327
    L9.Device_Model_N                 0.182817         0.090089            2.029           0.042
    L9.Content_TV_Show_N             -0.001668         0.003646           -0.457           0.647
    L9.Country_N                     -0.713921         1.171847           -0.609           0.542
    L9.City_N                         0.041020         0.021771            1.884           0.060
    L9.Region_N                       0.516188        21.902655            0.024           0.981
    ============================================================================================
    
    Results for equation Device_Vendor_N
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             5.280932         0.713867            7.398           0.000
    L1.Playtime                      -0.000003         0.000009           -0.353           0.724
    L1.Interruptions                  0.001004         0.001117            0.899           0.369
    L1.Join Time                      0.003584         0.005100            0.703           0.482
    L1.Buffer Ratio                  -0.002028         0.005059           -0.401           0.689
    L1.Connection Type                0.011918         0.004365            2.731           0.006
    L1.Device                        -0.103912         0.012463           -8.338           0.000
    L1.Device Type                    0.018867         0.010110            1.866           0.062
    L1.Browser                       -0.057115         0.014831           -3.851           0.000
    L1.OS                             0.328054         0.046537            7.049           0.000
    L1.OS Version                    -0.015029         0.004889           -3.074           0.002
    L1.Device ID                     -0.000007         0.000034           -0.214           0.830
    L1.Happiness Score               -0.010181         0.004145           -2.456           0.014
    L1.Playback Stalls                0.008390         0.073574            0.114           0.909
    L1.Startup Error (Count)         -0.932326         0.354391           -2.631           0.009
    L1.Latency                       -0.000000         0.000001           -0.329           0.742
    L1.Crash Status                   0.445841         0.216945            2.055           0.040
    L1.End of Playback Status        -0.155234         0.093464           -1.661           0.097
    L1.User_ID_N                     -0.000116         0.000109           -1.070           0.284
    L1.Title_N                        0.000105         0.000029            3.578           0.000
    L1.Device_Vendor_N                0.146979         0.006122           24.009           0.000
    L1.Device_Model_N                 0.000772         0.000722            1.070           0.285
    L1.Content_TV_Show_N              0.000262         0.000029            8.961           0.000
    L1.Country_N                      0.044792         0.009390            4.770           0.000
    L1.City_N                         0.001117         0.000174            6.409           0.000
    L1.Region_N                      -0.124629         0.175435           -0.710           0.477
    L2.Playtime                       0.000009         0.000009            1.063           0.288
    L2.Interruptions                  0.000536         0.001117            0.480           0.631
    L2.Join Time                      0.007415         0.005101            1.453           0.146
    L2.Buffer Ratio                  -0.007717         0.005072           -1.522           0.128
    L2.Connection Type                0.004286         0.004420            0.970           0.332
    L2.Device                        -0.055574         0.012584           -4.416           0.000
    L2.Device Type                    0.016979         0.010248            1.657           0.098
    L2.Browser                        0.018157         0.015044            1.207           0.227
    L2.OS                             0.104830         0.047044            2.228           0.026
    L2.OS Version                    -0.004588         0.004923           -0.932           0.351
    L2.Device ID                     -0.000063         0.000035           -1.823           0.068
    L2.Happiness Score               -0.010466         0.004199           -2.492           0.013
    L2.Playback Stalls               -0.051321         0.073579           -0.697           0.485
    L2.Startup Error (Count)         -0.775246         0.354595           -2.186           0.029
    L2.Latency                       -0.000000         0.000001           -0.323           0.747
    L2.Crash Status                   0.214167         0.217484            0.985           0.325
    L2.End of Playback Status        -0.212767         0.093669           -2.271           0.023
    L2.User_ID_N                     -0.000088         0.000110           -0.801           0.423
    L2.Title_N                        0.000005         0.000030            0.168           0.866
    L2.Device_Vendor_N                0.122506         0.006201           19.755           0.000
    L2.Device_Model_N                 0.000626         0.000730            0.856           0.392
    L2.Content_TV_Show_N              0.000063         0.000029            2.146           0.032
    L2.Country_N                     -0.013343         0.009458           -1.411           0.158
    L2.City_N                         0.000407         0.000177            2.301           0.021
    L2.Region_N                       0.616748         0.176632            3.492           0.000
    L3.Playtime                       0.000010         0.000009            1.147           0.252
    L3.Interruptions                 -0.000251         0.001117           -0.225           0.822
    L3.Join Time                     -0.000679         0.005101           -0.133           0.894
    L3.Buffer Ratio                   0.003504         0.005074            0.691           0.490
    L3.Connection Type                0.000310         0.004456            0.070           0.945
    L3.Device                        -0.052346         0.012648           -4.139           0.000
    L3.Device Type                    0.016890         0.010326            1.636           0.102
    L3.Browser                        0.019580         0.015129            1.294           0.196
    L3.OS                             0.098521         0.047212            2.087           0.037
    L3.OS Version                    -0.004887         0.004936           -0.990           0.322
    L3.Device ID                      0.000000         0.000035            0.003           0.998
    L3.Happiness Score               -0.007436         0.004212           -1.765           0.077
    L3.Playback Stalls               -0.143273         0.073583           -1.947           0.052
    L3.Startup Error (Count)         -0.599056         0.354601           -1.689           0.091
    L3.Latency                       -0.000000         0.000001           -0.142           0.887
    L3.Crash Status                   0.140819         0.217421            0.648           0.517
    L3.End of Playback Status        -0.332148         0.093766           -3.542           0.000
    L3.User_ID_N                     -0.000166         0.000111           -1.493           0.135
    L3.Title_N                       -0.000025         0.000030           -0.848           0.396
    L3.Device_Vendor_N                0.090398         0.006245           14.475           0.000
    L3.Device_Model_N                 0.001256         0.000734            1.711           0.087
    L3.Content_TV_Show_N             -0.000029         0.000029           -0.979           0.328
    L3.Country_N                     -0.027187         0.009512           -2.858           0.004
    L3.City_N                         0.000018         0.000178            0.099           0.921
    L3.Region_N                       0.832244         0.177355            4.693           0.000
    L4.Playtime                      -0.000011         0.000009           -1.289           0.197
    L4.Interruptions                  0.000993         0.001117            0.889           0.374
    L4.Join Time                      0.002362         0.005074            0.465           0.642
    L4.Buffer Ratio                  -0.011912         0.005073           -2.348           0.019
    L4.Connection Type               -0.006750         0.004473           -1.509           0.131
    L4.Device                         0.019355         0.012678            1.527           0.127
    L4.Device Type                   -0.000549         0.010363           -0.053           0.958
    L4.Browser                       -0.000097         0.015180           -0.006           0.995
    L4.OS                            -0.060183         0.047341           -1.271           0.204
    L4.OS Version                     0.007404         0.004948            1.496           0.135
    L4.Device ID                     -0.000039         0.000035           -1.113           0.266
    L4.Happiness Score                0.001989         0.004216            0.472           0.637
    L4.Playback Stalls                0.031682         0.073554            0.431           0.667
    L4.Startup Error (Count)         -0.122568         0.354583           -0.346           0.730
    L4.Latency                        0.000000         0.000001            0.180           0.857
    L4.Crash Status                  -0.281521         0.217420           -1.295           0.195
    L4.End of Playback Status        -0.076870         0.093787           -0.820           0.412
    L4.User_ID_N                     -0.000053         0.000112           -0.472           0.637
    L4.Title_N                        0.000031         0.000030            1.053           0.292
    L4.Device_Vendor_N                0.066056         0.006267           10.541           0.000
    L4.Device_Model_N                -0.001368         0.000736           -1.859           0.063
    L4.Content_TV_Show_N             -0.000046         0.000029           -1.551           0.121
    L4.Country_N                     -0.012118         0.009536           -1.271           0.204
    L4.City_N                        -0.000425         0.000179           -2.377           0.017
    L4.Region_N                       0.346048         0.177857            1.946           0.052
    L5.Playtime                       0.000013         0.000009            1.466           0.143
    L5.Interruptions                  0.000233         0.001117            0.208           0.835
    L5.Join Time                     -0.005041         0.005076           -0.993           0.321
    L5.Buffer Ratio                  -0.008816         0.005066           -1.740           0.082
    L5.Connection Type                0.000503         0.004477            0.112           0.911
    L5.Device                         0.021346         0.012687            1.683           0.092
    L5.Device Type                    0.004644         0.010376            0.448           0.654
    L5.Browser                       -0.003458         0.015204           -0.227           0.820
    L5.OS                            -0.150099         0.047388           -3.167           0.002
    L5.OS Version                     0.016328         0.004953            3.296           0.001
    L5.Device ID                     -0.000010         0.000035           -0.298           0.765
    L5.Happiness Score               -0.004622         0.004217           -1.096           0.273
    L5.Playback Stalls               -0.112148         0.071623           -1.566           0.117
    L5.Startup Error (Count)          0.065575         0.354550            0.185           0.853
    L5.Latency                        0.000000         0.000001            0.708           0.479
    L5.Crash Status                  -0.175368         0.217449           -0.806           0.420
    L5.End of Playback Status        -0.078795         0.093823           -0.840           0.401
    L5.User_ID_N                      0.000068         0.000112            0.610           0.542
    L5.Title_N                        0.000041         0.000030            1.372           0.170
    L5.Device_Vendor_N                0.062186         0.006271            9.916           0.000
    L5.Device_Model_N                 0.001984         0.000737            2.694           0.007
    L5.Content_TV_Show_N             -0.000058         0.000029           -1.955           0.051
    L5.Country_N                     -0.010733         0.009554           -1.123           0.261
    L5.City_N                        -0.000139         0.000179           -0.778           0.436
    L5.Region_N                       0.138511         0.178098            0.778           0.437
    L6.Playtime                       0.000001         0.000009            0.061           0.952
    L6.Interruptions                  0.000708         0.001117            0.634           0.526
    L6.Join Time                      0.004755         0.005078            0.936           0.349
    L6.Buffer Ratio                  -0.006544         0.005066           -1.292           0.196
    L6.Connection Type               -0.004677         0.004473           -1.045           0.296
    L6.Device                         0.009962         0.012679            0.786           0.432
    L6.Device Type                    0.022892         0.010366            2.208           0.027
    L6.Browser                        0.037439         0.015180            2.466           0.014
    L6.OS                             0.000058         0.047341            0.001           0.999
    L6.OS Version                    -0.011857         0.004949           -2.396           0.017
    L6.Device ID                      0.000012         0.000035            0.330           0.741
    L6.Happiness Score                0.000138         0.004214            0.033           0.974
    L6.Playback Stalls               -0.053495         0.073397           -0.729           0.466
    L6.Startup Error (Count)         -0.243145         0.354500           -0.686           0.493
    L6.Latency                       -0.000000         0.000001           -0.180           0.857
    L6.Crash Status                   0.139855         0.217434            0.643           0.520
    L6.End of Playback Status        -0.021821         0.093837           -0.233           0.816
    L6.User_ID_N                     -0.000031         0.000112           -0.274           0.784
    L6.Title_N                       -0.000033         0.000030           -1.120           0.263
    L6.Device_Vendor_N                0.022538         0.006266            3.597           0.000
    L6.Device_Model_N                -0.000963         0.000736           -1.308           0.191
    L6.Content_TV_Show_N             -0.000047         0.000029           -1.584           0.113
    L6.Country_N                     -0.024115         0.009534           -2.529           0.011
    L6.City_N                        -0.000275         0.000179           -1.540           0.124
    L6.Region_N                       0.681037         0.177833            3.830           0.000
    L7.Playtime                       0.000005         0.000009            0.631           0.528
    L7.Interruptions                  0.000284         0.001117            0.254           0.799
    L7.Join Time                      0.001820         0.005077            0.358           0.720
    L7.Buffer Ratio                   0.009412         0.005067            1.858           0.063
    L7.Connection Type               -0.010203         0.004456           -2.290           0.022
    L7.Device                         0.001765         0.012645            0.140           0.889
    L7.Device Type                    0.019602         0.010326            1.898           0.058
    L7.Browser                       -0.003687         0.015131           -0.244           0.807
    L7.OS                            -0.137252         0.047209           -2.907           0.004
    L7.OS Version                     0.016456         0.004936            3.334           0.001
    L7.Device ID                     -0.000032         0.000035           -0.910           0.363
    L7.Happiness Score                0.001353         0.004208            0.321           0.748
    L7.Playback Stalls               -0.126900         0.073778           -1.720           0.085
    L7.Startup Error (Count)         -0.488391         0.354470           -1.378           0.168
    L7.Latency                       -0.000002         0.000001           -2.902           0.004
    L7.Crash Status                   0.631424         0.217430            2.904           0.004
    L7.End of Playback Status         0.104090         0.093824            1.109           0.267
    L7.User_ID_N                      0.000070         0.000111            0.632           0.528
    L7.Title_N                        0.000027         0.000030            0.924           0.355
    L7.Device_Vendor_N                0.025466         0.006246            4.077           0.000
    L7.Device_Model_N                 0.000569         0.000734            0.775           0.438
    L7.Content_TV_Show_N             -0.000011         0.000029           -0.386           0.700
    L7.Country_N                     -0.010063         0.009509           -1.058           0.290
    L7.City_N                        -0.000105         0.000178           -0.588           0.557
    L7.Region_N                       0.198348         0.177345            1.118           0.263
    L8.Playtime                      -0.000013         0.000009           -1.537           0.124
    L8.Interruptions                 -0.000118         0.001117           -0.106           0.916
    L8.Join Time                     -0.002926         0.005077           -0.576           0.564
    L8.Buffer Ratio                   0.001023         0.005066            0.202           0.840
    L8.Connection Type                0.001957         0.004420            0.443           0.658
    L8.Device                         0.025346         0.012580            2.015           0.044
    L8.Device Type                   -0.010469         0.010249           -1.022           0.307
    L8.Browser                        0.015532         0.015051            1.032           0.302
    L8.OS                            -0.113200         0.047035           -2.407           0.016
    L8.OS Version                     0.008639         0.004922            1.755           0.079
    L8.Device ID                     -0.000041         0.000035           -1.187           0.235
    L8.Happiness Score               -0.003165         0.004197           -0.754           0.451
    L8.Playback Stalls               -0.026306         0.073780           -0.357           0.721
    L8.Startup Error (Count)         -0.195836         0.354218           -0.553           0.580
    L8.Latency                        0.000002         0.000001            3.230           0.001
    L8.Crash Status                   0.326513         0.217471            1.501           0.133
    L8.End of Playback Status        -0.023902         0.093801           -0.255           0.799
    L8.User_ID_N                      0.000045         0.000110            0.411           0.681
    L8.Title_N                       -0.000007         0.000030           -0.242           0.808
    L8.Device_Vendor_N                0.042737         0.006202            6.891           0.000
    L8.Device_Model_N                -0.000380         0.000730           -0.520           0.603
    L8.Content_TV_Show_N             -0.000041         0.000029           -1.404           0.160
    L8.Country_N                     -0.028584         0.009455           -3.023           0.003
    L8.City_N                        -0.000130         0.000177           -0.734           0.463
    L8.Region_N                       0.415348         0.176614            2.352           0.019
    L9.Playtime                      -0.000000         0.000009           -0.056           0.955
    L9.Interruptions                 -0.000165         0.001117           -0.148           0.883
    L9.Join Time                      0.002885         0.005074            0.569           0.570
    L9.Buffer Ratio                   0.007422         0.005065            1.465           0.143
    L9.Connection Type                0.002221         0.004365            0.509           0.611
    L9.Device                         0.017543         0.012455            1.408           0.159
    L9.Device Type                   -0.015457         0.010105           -1.530           0.126
    L9.Browser                        0.023677         0.014834            1.596           0.110
    L9.OS                            -0.097064         0.046486           -2.088           0.037
    L9.OS Version                     0.004926         0.004887            1.008           0.313
    L9.Device ID                     -0.000043         0.000034           -1.265           0.206
    L9.Happiness Score                0.003184         0.004140            0.769           0.442
    L9.Playback Stalls                0.171999         0.073524            2.339           0.019
    L9.Startup Error (Count)          1.060465         0.353899            2.997           0.003
    L9.Latency                       -0.000000         0.000001           -0.347           0.729
    L9.Crash Status                  -0.302123         0.217290           -1.390           0.164
    L9.End of Playback Status         0.177633         0.093767            1.894           0.058
    L9.User_ID_N                      0.000147         0.000109            1.351           0.177
    L9.Title_N                        0.000004         0.000029            0.148           0.882
    L9.Device_Vendor_N                0.034241         0.006122            5.593           0.000
    L9.Device_Model_N                 0.000086         0.000722            0.120           0.905
    L9.Content_TV_Show_N              0.000013         0.000029            0.438           0.661
    L9.Country_N                     -0.013379         0.009389           -1.425           0.154
    L9.City_N                         0.000250         0.000174            1.431           0.152
    L9.Region_N                       0.172349         0.175483            0.982           0.326
    ============================================================================================
    
    Results for equation Device_Model_N
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             7.210539         5.227407            1.379           0.168
    L1.Playtime                      -0.000051         0.000064           -0.807           0.420
    L1.Interruptions                 -0.011108         0.008179           -1.358           0.174
    L1.Join Time                      0.045494         0.037345            1.218           0.223
    L1.Buffer Ratio                   0.023763         0.037049            0.641           0.521
    L1.Connection Type                0.041780         0.031960            1.307           0.191
    L1.Device                        -0.191875         0.091261           -2.102           0.036
    L1.Device Type                   -0.001567         0.074035           -0.021           0.983
    L1.Browser                        0.407727         0.108606            3.754           0.000
    L1.OS                             0.525528         0.340771            1.542           0.123
    L1.OS Version                    -0.003133         0.035799           -0.088           0.930
    L1.Device ID                     -0.000125         0.000250           -0.499           0.618
    L1.Happiness Score                0.012280         0.030351            0.405           0.686
    L1.Playback Stalls               -0.305935         0.538758           -0.568           0.570
    L1.Startup Error (Count)          4.761870         2.595086            1.835           0.067
    L1.Latency                        0.000008         0.000005            1.765           0.078
    L1.Crash Status                  -7.272816         1.588614           -4.578           0.000
    L1.End of Playback Status         1.331629         0.684405            1.946           0.052
    L1.User_ID_N                      0.001330         0.000796            1.671           0.095
    L1.Title_N                        0.000027         0.000216            0.124           0.901
    L1.Device_Vendor_N                0.175216         0.044828            3.909           0.000
    L1.Device_Model_N                 0.159648         0.005286           30.205           0.000
    L1.Content_TV_Show_N             -0.001275         0.000214           -5.957           0.000
    L1.Country_N                     -0.424175         0.068761           -6.169           0.000
    L1.City_N                        -0.004506         0.001276           -3.531           0.000
    L1.Region_N                       4.567040         1.284652            3.555           0.000
    L2.Playtime                      -0.000086         0.000064           -1.350           0.177
    L2.Interruptions                 -0.001297         0.008178           -0.159           0.874
    L2.Join Time                      0.027663         0.037355            0.741           0.459
    L2.Buffer Ratio                   0.025351         0.037137            0.683           0.495
    L2.Connection Type                0.008136         0.032365            0.251           0.802
    L2.Device                        -0.088695         0.092149           -0.963           0.336
    L2.Device Type                   -0.162074         0.075039           -2.160           0.031
    L2.Browser                       -0.197186         0.110163           -1.790           0.073
    L2.OS                             0.362307         0.344488            1.052           0.293
    L2.OS Version                     0.007236         0.036047            0.201           0.841
    L2.Device ID                      0.000112         0.000253            0.443           0.658
    L2.Happiness Score                0.009578         0.030751            0.311           0.755
    L2.Playback Stalls                0.276137         0.538795            0.513           0.608
    L2.Startup Error (Count)          2.375329         2.596577            0.915           0.360
    L2.Latency                        0.000004         0.000005            0.841           0.400
    L2.Crash Status                   0.009507         1.592564            0.006           0.995
    L2.End of Playback Status         1.021226         0.685909            1.489           0.137
    L2.User_ID_N                      0.000650         0.000808            0.804           0.421
    L2.Title_N                        0.000045         0.000216            0.208           0.835
    L2.Device_Vendor_N                0.089462         0.045411            1.970           0.049
    L2.Device_Model_N                 0.116145         0.005349           21.715           0.000
    L2.Content_TV_Show_N              0.000070         0.000215            0.324           0.746
    L2.Country_N                      0.257273         0.069255            3.715           0.000
    L2.City_N                        -0.001884         0.001296           -1.454           0.146
    L2.Region_N                      -3.492641         1.293419           -2.700           0.007
    L3.Playtime                       0.000001         0.000064            0.015           0.988
    L3.Interruptions                 -0.002634         0.008177           -0.322           0.747
    L3.Join Time                     -0.019131         0.037354           -0.512           0.609
    L3.Buffer Ratio                   0.023207         0.037152            0.625           0.532
    L3.Connection Type                0.010216         0.032628            0.313           0.754
    L3.Device                        -0.019983         0.092620           -0.216           0.829
    L3.Device Type                   -0.073088         0.075615           -0.967           0.334
    L3.Browser                       -0.013084         0.110782           -0.118           0.906
    L3.OS                             0.200081         0.345719            0.579           0.563
    L3.OS Version                     0.015535         0.036141            0.430           0.667
    L3.Device ID                      0.000017         0.000255            0.066           0.947
    L3.Happiness Score                0.045379         0.030845            1.471           0.141
    L3.Playback Stalls                1.106768         0.538825            2.054           0.040
    L3.Startup Error (Count)          3.912689         2.596621            1.507           0.132
    L3.Latency                        0.000002         0.000005            0.389           0.697
    L3.Crash Status                  -0.751998         1.592098           -0.472           0.637
    L3.End of Playback Status         2.110104         0.686619            3.073           0.002
    L3.User_ID_N                      0.000896         0.000814            1.100           0.271
    L3.Title_N                        0.000183         0.000216            0.845           0.398
    L3.Device_Vendor_N                0.009057         0.045730            0.198           0.843
    L3.Device_Model_N                 0.081754         0.005377           15.205           0.000
    L3.Content_TV_Show_N              0.000623         0.000215            2.893           0.004
    L3.Country_N                      0.157284         0.069652            2.258           0.024
    L3.City_N                        -0.000870         0.001306           -0.666           0.505
    L3.Region_N                      -3.512921         1.298707           -2.705           0.007
    L4.Playtime                      -0.000010         0.000064           -0.161           0.872
    L4.Interruptions                 -0.008046         0.008177           -0.984           0.325
    L4.Join Time                     -0.020771         0.037155           -0.559           0.576
    L4.Buffer Ratio                   0.053175         0.037149            1.431           0.152
    L4.Connection Type                0.036447         0.032757            1.113           0.266
    L4.Device                        -0.192809         0.092834           -2.077           0.038
    L4.Device Type                    0.082131         0.075884            1.082           0.279
    L4.Browser                       -0.211525         0.111161           -1.903           0.057
    L4.OS                             0.495932         0.346661            1.431           0.153
    L4.OS Version                    -0.016191         0.036235           -0.447           0.655
    L4.Device ID                      0.000090         0.000256            0.351           0.725
    L4.Happiness Score                0.016420         0.030876            0.532           0.595
    L4.Playback Stalls                0.046761         0.538609            0.087           0.931
    L4.Startup Error (Count)         -2.607425         2.596493           -1.004           0.315
    L4.Latency                       -0.000003         0.000005           -0.658           0.510
    L4.Crash Status                   3.283053         1.592091            2.062           0.039
    L4.End of Playback Status        -0.153653         0.686767           -0.224           0.823
    L4.User_ID_N                     -0.000221         0.000817           -0.271           0.787
    L4.Title_N                       -0.000084         0.000216           -0.390           0.697
    L4.Device_Vendor_N               -0.032184         0.045888           -0.701           0.483
    L4.Device_Model_N                 0.071449         0.005390           13.256           0.000
    L4.Content_TV_Show_N              0.000600         0.000216            2.785           0.005
    L4.Country_N                      0.263677         0.069828            3.776           0.000
    L4.City_N                         0.001494         0.001310            1.141           0.254
    L4.Region_N                      -4.079347         1.302386           -3.132           0.002
    L5.Playtime                      -0.000026         0.000064           -0.414           0.679
    L5.Interruptions                  0.000681         0.008178            0.083           0.934
    L5.Join Time                      0.004518         0.037169            0.122           0.903
    L5.Buffer Ratio                   0.044177         0.037095            1.191           0.234
    L5.Connection Type               -0.002409         0.032785           -0.073           0.941
    L5.Device                        -0.260393         0.092899           -2.803           0.005
    L5.Device Type                    0.247369         0.075976            3.256           0.001
    L5.Browser                       -0.140403         0.111333           -1.261           0.207
    L5.OS                             1.608508         0.347007            4.635           0.000
    L5.OS Version                    -0.113395         0.036271           -3.126           0.002
    L5.Device ID                     -0.000146         0.000256           -0.569           0.569
    L5.Happiness Score                0.014948         0.030880            0.484           0.628
    L5.Playback Stalls                1.221143         0.524474            2.328           0.020
    L5.Startup Error (Count)          1.892692         2.596249            0.729           0.466
    L5.Latency                       -0.000004         0.000005           -0.771           0.441
    L5.Crash Status                  -0.470267         1.592306           -0.295           0.768
    L5.End of Playback Status         0.844615         0.687037            1.229           0.219
    L5.User_ID_N                     -0.001477         0.000818           -1.807           0.071
    L5.Title_N                       -0.000095         0.000216           -0.437           0.662
    L5.Device_Vendor_N                0.002812         0.045921            0.061           0.951
    L5.Device_Model_N                 0.044947         0.005393            8.334           0.000
    L5.Content_TV_Show_N              0.000249         0.000216            1.155           0.248
    L5.Country_N                      0.089204         0.069958            1.275           0.202
    L5.City_N                         0.001927         0.001311            1.469           0.142
    L5.Region_N                      -0.257695         1.304153           -0.198           0.843
    L6.Playtime                       0.000008         0.000064            0.122           0.903
    L6.Interruptions                 -0.008292         0.008178           -1.014           0.311
    L6.Join Time                     -0.032402         0.037184           -0.871           0.384
    L6.Buffer Ratio                  -0.002187         0.037097           -0.059           0.953
    L6.Connection Type                0.041604         0.032758            1.270           0.204
    L6.Device                        -0.153212         0.092843           -1.650           0.099
    L6.Device Type                   -0.036983         0.075905           -0.487           0.626
    L6.Browser                       -0.394431         0.111155           -3.548           0.000
    L6.OS                            -0.196289         0.346663           -0.566           0.571
    L6.OS Version                     0.103588         0.036237            2.859           0.004
    L6.Device ID                     -0.000252         0.000256           -0.984           0.325
    L6.Happiness Score               -0.032363         0.030857           -1.049           0.294
    L6.Playback Stalls                0.580476         0.537459            1.080           0.280
    L6.Startup Error (Count)         -1.800196         2.595886           -0.693           0.488
    L6.Latency                       -0.000002         0.000005           -0.491           0.623
    L6.Crash Status                   1.086453         1.592192            0.682           0.495
    L6.End of Playback Status        -0.396496         0.687136           -0.577           0.564
    L6.User_ID_N                      0.000826         0.000817            1.011           0.312
    L6.Title_N                        0.000348         0.000216            1.607           0.108
    L6.Device_Vendor_N               -0.054337         0.045883           -1.184           0.236
    L6.Device_Model_N                 0.052474         0.005389            9.737           0.000
    L6.Content_TV_Show_N             -0.000190         0.000216           -0.882           0.378
    L6.Country_N                      0.114746         0.069817            1.644           0.100
    L6.City_N                         0.001053         0.001310            0.804           0.421
    L6.Region_N                      -5.487472         1.302211           -4.214           0.000
    L7.Playtime                      -0.000043         0.000064           -0.680           0.497
    L7.Interruptions                  0.001848         0.008178            0.226           0.821
    L7.Join Time                     -0.037301         0.037180           -1.003           0.316
    L7.Buffer Ratio                  -0.007793         0.037103           -0.210           0.834
    L7.Connection Type                0.008729         0.032627            0.268           0.789
    L7.Device                        -0.146469         0.092592           -1.582           0.114
    L7.Device Type                   -0.006625         0.075617           -0.088           0.930
    L7.Browser                       -0.060544         0.110799           -0.546           0.585
    L7.OS                             0.565825         0.345695            1.637           0.102
    L7.OS Version                    -0.061340         0.036148           -1.697           0.090
    L7.Device ID                      0.000101         0.000255            0.396           0.692
    L7.Happiness Score               -0.020146         0.030816           -0.654           0.513
    L7.Playback Stalls                0.665312         0.540253            1.231           0.218
    L7.Startup Error (Count)         -0.385249         2.595666           -0.148           0.882
    L7.Latency                        0.000010         0.000005            2.088           0.037
    L7.Crash Status                  -0.875685         1.592163           -0.550           0.582
    L7.End of Playback Status        -0.923962         0.687042           -1.345           0.179
    L7.User_ID_N                      0.000552         0.000814            0.678           0.498
    L7.Title_N                        0.000286         0.000216            1.320           0.187
    L7.Device_Vendor_N               -0.089259         0.045736           -1.952           0.051
    L7.Device_Model_N                 0.040351         0.005376            7.506           0.000
    L7.Content_TV_Show_N             -0.000151         0.000215           -0.699           0.484
    L7.Country_N                      0.100082         0.069629            1.437           0.151
    L7.City_N                        -0.000828         0.001306           -0.634           0.526
    L7.Region_N                      -1.259847         1.298635           -0.970           0.332
    L8.Playtime                       0.000013         0.000064            0.202           0.840
    L8.Interruptions                 -0.000828         0.008178           -0.101           0.919
    L8.Join Time                      0.012728         0.037176            0.342           0.732
    L8.Buffer Ratio                  -0.018356         0.037097           -0.495           0.621
    L8.Connection Type               -0.009705         0.032368           -0.300           0.764
    L8.Device                         0.066438         0.092120            0.721           0.471
    L8.Device Type                    0.016665         0.075048            0.222           0.824
    L8.Browser                       -0.219477         0.110210           -1.991           0.046
    L8.OS                             0.193567         0.344418            0.562           0.574
    L8.OS Version                    -0.029632         0.036041           -0.822           0.411
    L8.Device ID                      0.000393         0.000253            1.552           0.121
    L8.Happiness Score                0.012234         0.030733            0.398           0.691
    L8.Playback Stalls                0.553360         0.540264            1.024           0.306
    L8.Startup Error (Count)          3.091505         2.593819            1.192           0.233
    L8.Latency                       -0.000011         0.000005           -2.263           0.024
    L8.Crash Status                  -1.149744         1.592464           -0.722           0.470
    L8.End of Playback Status         1.090234         0.686873            1.587           0.112
    L8.User_ID_N                      0.000077         0.000808            0.095           0.924
    L8.Title_N                        0.000113         0.000216            0.525           0.600
    L8.Device_Vendor_N               -0.075787         0.045414           -1.669           0.095
    L8.Device_Model_N                 0.034004         0.005348            6.359           0.000
    L8.Content_TV_Show_N              0.000029         0.000215            0.134           0.893
    L8.Country_N                      0.077932         0.069234            1.126           0.260
    L8.City_N                         0.001072         0.001296            0.828           0.408
    L8.Region_N                      -1.754666         1.293280           -1.357           0.175
    L9.Playtime                       0.000039         0.000064            0.617           0.537
    L9.Interruptions                 -0.002324         0.008178           -0.284           0.776
    L9.Join Time                      0.021487         0.037155            0.578           0.563
    L9.Buffer Ratio                  -0.015333         0.037090           -0.413           0.679
    L9.Connection Type                0.002942         0.031963            0.092           0.927
    L9.Device                        -0.166653         0.091203           -1.827           0.068
    L9.Device Type                    0.122854         0.073994            1.660           0.097
    L9.Browser                       -0.209733         0.108624           -1.931           0.054
    L9.OS                             0.847567         0.340399            2.490           0.013
    L9.OS Version                    -0.044501         0.035786           -1.244           0.214
    L9.Device ID                      0.000416         0.000251            1.659           0.097
    L9.Happiness Score               -0.032021         0.030316           -1.056           0.291
    L9.Playback Stalls               -0.813426         0.538392           -1.511           0.131
    L9.Startup Error (Count)          1.355002         2.591482            0.523           0.601
    L9.Latency                        0.000003         0.000005            0.634           0.526
    L9.Crash Status                   0.154660         1.591137            0.097           0.923
    L9.End of Playback Status         0.658822         0.686626            0.960           0.337
    L9.User_ID_N                     -0.000686         0.000796           -0.861           0.389
    L9.Title_N                       -0.000012         0.000216           -0.054           0.957
    L9.Device_Vendor_N               -0.025990         0.044828           -0.580           0.562
    L9.Device_Model_N                 0.038335         0.005285            7.253           0.000
    L9.Content_TV_Show_N              0.000115         0.000214            0.537           0.592
    L9.Country_N                      0.132715         0.068751            1.930           0.054
    L9.City_N                        -0.000016         0.001277           -0.013           0.990
    L9.Region_N                      -1.462596         1.284999           -1.138           0.255
    ============================================================================================
    
    Results for equation Content_TV_Show_N
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                          1381.207636       100.553153           13.736           0.000
    L1.Playtime                       0.001057         0.001226            0.862           0.389
    L1.Interruptions                  0.074410         0.157334            0.473           0.636
    L1.Join Time                      1.043739         0.718354            1.453           0.146
    L1.Buffer Ratio                  -0.864989         0.712664           -1.214           0.225
    L1.Connection Type               -0.077865         0.614778           -0.127           0.899
    L1.Device                        -9.229044         1.755479           -5.257           0.000
    L1.Device Type                    0.078341         1.424119            0.055           0.956
    L1.Browser                       -5.969655         2.089114           -2.858           0.004
    L1.OS                            23.361586         6.554991            3.564           0.000
    L1.OS Version                    -1.330514         0.688615           -1.932           0.053
    L1.Device ID                      0.001142         0.004816            0.237           0.813
    L1.Happiness Score               -0.367819         0.583831           -0.630           0.529
    L1.Playback Stalls               16.391833        10.363423            1.582           0.114
    L1.Startup Error (Count)        -74.073726        49.918462           -1.484           0.138
    L1.Latency                       -0.000061         0.000090           -0.672           0.501
    L1.Crash Status                  20.417750        30.558199            0.668           0.504
    L1.End of Playback Status       -14.026054        13.165058           -1.065           0.287
    L1.User_ID_N                     -0.031240         0.015312           -2.040           0.041
    L1.Title_N                       -0.004244         0.004153           -1.022           0.307
    L1.Device_Vendor_N                1.460114         0.862304            1.693           0.090
    L1.Device_Model_N                -0.422496         0.101671           -4.156           0.000
    L1.Content_TV_Show_N              0.101097         0.004116           24.562           0.000
    L1.Country_N                     -1.527760         1.322677           -1.155           0.248
    L1.City_N                         0.116147         0.024549            4.731           0.000
    L1.Region_N                      64.089677        24.711261            2.594           0.009
    L2.Playtime                      -0.001259         0.001226           -1.027           0.304
    L2.Interruptions                  0.093134         0.157301            0.592           0.554
    L2.Join Time                     -0.092375         0.718559           -0.129           0.898
    L2.Buffer Ratio                   0.881863         0.714361            1.234           0.217
    L2.Connection Type                1.541035         0.622566            2.475           0.013
    L2.Device                        -1.376605         1.772551           -0.777           0.437
    L2.Device Type                    5.643284         1.443434            3.910           0.000
    L2.Browser                       -4.937069         2.119062           -2.330           0.020
    L2.OS                             4.817423         6.626496            0.727           0.467
    L2.OS Version                     0.408445         0.693390            0.589           0.556
    L2.Device ID                      0.001140         0.004873            0.234           0.815
    L2.Happiness Score               -0.728536         0.591525           -1.232           0.218
    L2.Playback Stalls              -23.089148        10.364134           -2.228           0.026
    L2.Startup Error (Count)        -43.979058        49.947135           -0.881           0.379
    L2.Latency                       -0.000068         0.000090           -0.760           0.447
    L2.Crash Status                 -38.752211        30.634182           -1.265           0.206
    L2.End of Playback Status        -1.258485        13.193982           -0.095           0.924
    L2.User_ID_N                     -0.022362         0.015549           -1.438           0.150
    L2.Title_N                        0.004308         0.004159            1.036           0.300
    L2.Device_Vendor_N                2.960943         0.873508            3.390           0.001
    L2.Device_Model_N                -0.036634         0.102884           -0.356           0.722
    L2.Content_TV_Show_N              0.070814         0.004133           17.134           0.000
    L2.Country_N                     -0.937586         1.332167           -0.704           0.482
    L2.City_N                         0.071705         0.024924            2.877           0.004
    L2.Region_N                       2.566042        24.879894            0.103           0.918
    L3.Playtime                       0.003199         0.001226            2.609           0.009
    L3.Interruptions                 -0.034379         0.157298           -0.219           0.827
    L3.Join Time                     -1.010087         0.718535           -1.406           0.160
    L3.Buffer Ratio                  -2.309492         0.714641           -3.232           0.001
    L3.Connection Type               -1.151287         0.627627           -1.834           0.067
    L3.Device                        -6.507692         1.781613           -3.653           0.000
    L3.Device Type                    3.604306         1.454506            2.478           0.013
    L3.Browser                        3.325522         2.130969            1.561           0.119
    L3.OS                            10.220214         6.650174            1.537           0.124
    L3.OS Version                    -1.591889         0.695208           -2.290           0.022
    L3.Device ID                     -0.000018         0.004910           -0.004           0.997
    L3.Happiness Score               -0.916080         0.593326           -1.544           0.123
    L3.Playback Stalls                5.872875        10.364707            0.567           0.571
    L3.Startup Error (Count)        -62.806226        49.947986           -1.257           0.209
    L3.Latency                        0.000046         0.000090            0.507           0.612
    L3.Crash Status                 -14.703245        30.625217           -0.480           0.631
    L3.End of Playback Status       -20.824969        13.207643           -1.577           0.115
    L3.User_ID_N                     -0.008098         0.015662           -0.517           0.605
    L3.Title_N                        0.002954         0.004162            0.710           0.478
    L3.Device_Vendor_N               -2.587376         0.879655           -2.941           0.003
    L3.Device_Model_N                -0.001673         0.103423           -0.016           0.987
    L3.Content_TV_Show_N              0.061011         0.004143           14.728           0.000
    L3.Country_N                     -4.743408         1.339809           -3.540           0.000
    L3.City_N                        -0.050677         0.025113           -2.018           0.044
    L3.Region_N                     105.148882        24.981629            4.209           0.000
    L4.Playtime                       0.000861         0.001226            0.702           0.482
    L4.Interruptions                  0.024981         0.157297            0.159           0.874
    L4.Join Time                     -0.371402         0.714710           -0.520           0.603
    L4.Buffer Ratio                   0.767472         0.714591            1.074           0.283
    L4.Connection Type               -0.300643         0.630115           -0.477           0.633
    L4.Device                        -1.203831         1.785737           -0.674           0.500
    L4.Device Type                    2.043096         1.459680            1.400           0.162
    L4.Browser                       -2.824994         2.138258           -1.321           0.186
    L4.OS                            -2.150971         6.668284           -0.323           0.747
    L4.OS Version                     0.689363         0.697001            0.989           0.323
    L4.Device ID                     -0.000722         0.004926           -0.147           0.883
    L4.Happiness Score                0.643270         0.593920            1.083           0.279
    L4.Playback Stalls               -8.566524        10.360546           -0.827           0.408
    L4.Startup Error (Count)         85.014498        49.945520            1.702           0.089
    L4.Latency                        0.000070         0.000090            0.777           0.437
    L4.Crash Status                 -81.733335        30.625078           -2.669           0.008
    L4.End of Playback Status        20.282866        13.210480            1.535           0.125
    L4.User_ID_N                     -0.013408         0.015713           -0.853           0.393
    L4.Title_N                       -0.003511         0.004160           -0.844           0.399
    L4.Device_Vendor_N               -0.655788         0.882688           -0.743           0.458
    L4.Device_Model_N                 0.141192         0.103682            1.362           0.173
    L4.Content_TV_Show_N              0.054199         0.004147           13.070           0.000
    L4.Country_N                     -0.428698         1.343197           -0.319           0.750
    L4.City_N                         0.010164         0.025192            0.403           0.687
    L4.Region_N                       7.809937        25.052382            0.312           0.755
    L5.Playtime                      -0.000829         0.001227           -0.676           0.499
    L5.Interruptions                  0.046640         0.157302            0.296           0.767
    L5.Join Time                     -0.132059         0.714983           -0.185           0.853
    L5.Buffer Ratio                  -0.230825         0.713555           -0.323           0.746
    L5.Connection Type                0.364003         0.630644            0.577           0.564
    L5.Device                         0.631701         1.786990            0.353           0.724
    L5.Device Type                   -0.079510         1.461465           -0.054           0.957
    L5.Browser                        0.878678         2.141583            0.410           0.682
    L5.OS                           -12.236926         6.674938           -1.833           0.067
    L5.OS Version                     0.534994         0.697707            0.767           0.443
    L5.Device ID                      0.003320         0.004929            0.674           0.501
    L5.Happiness Score                0.401694         0.593997            0.676           0.499
    L5.Playback Stalls               -2.948881        10.088649           -0.292           0.770
    L5.Startup Error (Count)          2.235536        49.940827            0.045           0.964
    L5.Latency                       -0.000001         0.000090           -0.012           0.990
    L5.Crash Status                 -29.625871        30.629226           -0.967           0.333
    L5.End of Playback Status        -0.549496        13.215677           -0.042           0.967
    L5.User_ID_N                     -0.001808         0.015727           -0.115           0.908
    L5.Title_N                        0.003098         0.004160            0.745           0.457
    L5.Device_Vendor_N               -1.689361         0.883320           -1.913           0.056
    L5.Device_Model_N                 0.205680         0.103747            1.983           0.047
    L5.Content_TV_Show_N              0.047762         0.004149           11.511           0.000
    L5.Country_N                     -2.226813         1.345697           -1.655           0.098
    L5.City_N                        -0.021178         0.025225           -0.840           0.401
    L5.Region_N                      32.527673        25.086372            1.297           0.195
    L6.Playtime                       0.000148         0.001227            0.121           0.904
    L6.Interruptions                  0.024110         0.157302            0.153           0.878
    L6.Join Time                      0.133301         0.715268            0.186           0.852
    L6.Buffer Ratio                   1.058013         0.713584            1.483           0.138
    L6.Connection Type               -0.494857         0.630120           -0.785           0.432
    L6.Device                         1.990872         1.785903            1.115           0.265
    L6.Device Type                   -2.073363         1.460092           -1.420           0.156
    L6.Browser                        7.085377         2.138151            3.314           0.001
    L6.OS                           -11.908281         6.668321           -1.786           0.074
    L6.OS Version                    -0.838313         0.697056           -1.203           0.229
    L6.Device ID                      0.003584         0.004926            0.728           0.467
    L6.Happiness Score                0.339840         0.593567            0.573           0.567
    L6.Playback Stalls              -33.895443        10.338427           -3.279           0.001
    L6.Startup Error (Count)         35.410028        49.933842            0.709           0.478
    L6.Latency                        0.000040         0.000090            0.448           0.654
    L6.Crash Status                  11.219363        30.627019            0.366           0.714
    L6.End of Playback Status        12.642150        13.217594            0.956           0.339
    L6.User_ID_N                      0.011299         0.015713            0.719           0.472
    L6.Title_N                       -0.002561         0.004161           -0.615           0.538
    L6.Device_Vendor_N               -3.754248         0.882593           -4.254           0.000
    L6.Device_Model_N                -0.059425         0.103663           -0.573           0.566
    L6.Content_TV_Show_N              0.048726         0.004148           11.748           0.000
    L6.Country_N                     -1.562013         1.342976           -1.163           0.245
    L6.City_N                        -0.018825         0.025193           -0.747           0.455
    L6.Region_N                      42.504159        25.049030            1.697           0.090
    L7.Playtime                       0.000803         0.001226            0.655           0.512
    L7.Interruptions                  0.042248         0.157308            0.269           0.788
    L7.Join Time                     -0.229040         0.715183           -0.320           0.749
    L7.Buffer Ratio                   0.329765         0.713704            0.462           0.644
    L7.Connection Type               -0.539519         0.627613           -0.860           0.390
    L7.Device                         0.925032         1.781078            0.519           0.604
    L7.Device Type                    5.796058         1.454549            3.985           0.000
    L7.Browser                       -3.955762         2.131301           -1.856           0.063
    L7.OS                             1.522404         6.649698            0.229           0.819
    L7.OS Version                     0.272549         0.695330            0.392           0.695
    L7.Device ID                      0.003312         0.004909            0.675           0.500
    L7.Happiness Score               -0.428681         0.592767           -0.723           0.470
    L7.Playback Stalls               -9.023417        10.392174           -0.868           0.385
    L7.Startup Error (Count)        -14.770555        49.929605           -0.296           0.767
    L7.Latency                        0.000047         0.000090            0.521           0.602
    L7.Crash Status                  19.837105        30.626463            0.648           0.517
    L7.End of Playback Status         6.616512        13.215776            0.501           0.617
    L7.User_ID_N                      0.006824         0.015662            0.436           0.663
    L7.Title_N                       -0.005023         0.004162           -1.207           0.227
    L7.Device_Vendor_N               -0.994611         0.879770           -1.131           0.258
    L7.Device_Model_N                -0.070775         0.103404           -0.684           0.494
    L7.Content_TV_Show_N              0.035038         0.004143            8.457           0.000
    L7.Country_N                     -0.544790         1.339367           -0.407           0.684
    L7.City_N                        -0.025501         0.025114           -1.015           0.310
    L7.Region_N                      39.188043        24.980234            1.569           0.117
    L8.Playtime                       0.000158         0.001226            0.128           0.898
    L8.Interruptions                  0.047486         0.157304            0.302           0.763
    L8.Join Time                     -1.376124         0.715110           -1.924           0.054
    L8.Buffer Ratio                  -0.379205         0.713592           -0.531           0.595
    L8.Connection Type               -0.464054         0.622625           -0.745           0.456
    L8.Device                        -1.555077         1.771992           -0.878           0.380
    L8.Device Type                    0.363133         1.443598            0.252           0.801
    L8.Browser                        4.651754         2.119982            2.194           0.028
    L8.OS                            -9.748731         6.625148           -1.471           0.141
    L8.OS Version                     0.499225         0.693280            0.720           0.471
    L8.Device ID                     -0.016211         0.004872           -3.327           0.001
    L8.Happiness Score               -0.744468         0.591171           -1.259           0.208
    L8.Playback Stalls               -1.208204        10.392395           -0.116           0.907
    L8.Startup Error (Count)        -91.330386        49.894088           -1.830           0.067
    L8.Latency                        0.000068         0.000090            0.753           0.451
    L8.Crash Status                  37.250944        30.632256            1.216           0.224
    L8.End of Playback Status       -14.731413        13.212532           -1.115           0.265
    L8.User_ID_N                      0.005850         0.015549            0.376           0.707
    L8.Title_N                       -0.000227         0.004158           -0.055           0.956
    L8.Device_Vendor_N               -1.537064         0.873570           -1.760           0.078
    L8.Device_Model_N                 0.221756         0.102869            2.156           0.031
    L8.Content_TV_Show_N              0.043568         0.004132           10.545           0.000
    L8.Country_N                     -0.089194         1.331761           -0.067           0.947
    L8.City_N                        -0.028500         0.024920           -1.144           0.253
    L8.Region_N                      18.379003        24.877234            0.739           0.460
    L9.Playtime                       0.000226         0.001226            0.185           0.854
    L9.Interruptions                  0.031327         0.157305            0.199           0.842
    L9.Join Time                     -0.487251         0.714710           -0.682           0.495
    L9.Buffer Ratio                   0.348338         0.713445            0.488           0.625
    L9.Connection Type                0.434339         0.614838            0.706           0.480
    L9.Device                        -0.583754         1.754357           -0.333           0.739
    L9.Device Type                    0.740554         1.423340            0.520           0.603
    L9.Browser                        0.363916         2.089459            0.174           0.862
    L9.OS                            -1.771197         6.547828           -0.271           0.787
    L9.OS Version                     0.533554         0.688379            0.775           0.438
    L9.Device ID                     -0.001603         0.004819           -0.333           0.739
    L9.Happiness Score                0.396983         0.583148            0.681           0.496
    L9.Playback Stalls               15.459608        10.356371            1.493           0.135
    L9.Startup Error (Count)         48.898348        49.849122            0.981           0.327
    L9.Latency                        0.000131         0.000090            1.453           0.146
    L9.Crash Status                 -31.098173        30.606733           -1.016           0.310
    L9.End of Playback Status        16.363524        13.207781            1.239           0.215
    L9.User_ID_N                     -0.018140         0.015321           -1.184           0.236
    L9.Title_N                       -0.004715         0.004151           -1.136           0.256
    L9.Device_Vendor_N               -0.083480         0.862310           -0.097           0.923
    L9.Device_Model_N                 0.070252         0.101669            0.691           0.490
    L9.Content_TV_Show_N              0.042726         0.004114           10.385           0.000
    L9.Country_N                     -1.384954         1.322471           -1.047           0.295
    L9.City_N                        -0.005627         0.024569           -0.229           0.819
    L9.Region_N                      42.973638        24.717928            1.739           0.082
    ============================================================================================
    
    Results for equation Country_N
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             1.189869         0.397946            2.990           0.003
    L1.Playtime                       0.000008         0.000005            1.569           0.117
    L1.Interruptions                 -0.000297         0.000623           -0.477           0.633
    L1.Join Time                      0.001631         0.002843            0.574           0.566
    L1.Buffer Ratio                   0.000474         0.002820            0.168           0.866
    L1.Connection Type               -0.002687         0.002433           -1.104           0.269
    L1.Device                         0.008342         0.006947            1.201           0.230
    L1.Device Type                    0.003996         0.005636            0.709           0.478
    L1.Browser                        0.011526         0.008268            1.394           0.163
    L1.OS                            -0.066528         0.025942           -2.565           0.010
    L1.OS Version                     0.006093         0.002725            2.236           0.025
    L1.Device ID                     -0.000042         0.000019           -2.197           0.028
    L1.Happiness Score                0.004776         0.002311            2.067           0.039
    L1.Playback Stalls               -0.012727         0.041014           -0.310           0.756
    L1.Startup Error (Count)          0.337397         0.197556            1.708           0.088
    L1.Latency                        0.000000         0.000000            1.070           0.285
    L1.Crash Status                  -0.378307         0.120936           -3.128           0.002
    L1.End of Playback Status         0.004015         0.052102            0.077           0.939
    L1.User_ID_N                      0.000062         0.000061            1.022           0.307
    L1.Title_N                       -0.000040         0.000016           -2.419           0.016
    L1.Device_Vendor_N                0.001622         0.003413            0.475           0.635
    L1.Device_Model_N                 0.000399         0.000402            0.992           0.321
    L1.Content_TV_Show_N             -0.000068         0.000016           -4.147           0.000
    L1.Country_N                      0.108863         0.005235           20.797           0.000
    L1.City_N                        -0.000294         0.000097           -3.027           0.002
    L1.Region_N                       0.419930         0.097796            4.294           0.000
    L2.Playtime                       0.000001         0.000005            0.248           0.804
    L2.Interruptions                 -0.000117         0.000623           -0.188           0.851
    L2.Join Time                      0.000397         0.002844            0.140           0.889
    L2.Buffer Ratio                   0.005407         0.002827            1.913           0.056
    L2.Connection Type                0.000366         0.002464            0.149           0.882
    L2.Device                         0.007863         0.007015            1.121           0.262
    L2.Device Type                   -0.016948         0.005712           -2.967           0.003
    L2.Browser                        0.000491         0.008386            0.059           0.953
    L2.OS                            -0.048610         0.026225           -1.854           0.064
    L2.OS Version                     0.003725         0.002744            1.358           0.175
    L2.Device ID                     -0.000006         0.000019           -0.298           0.766
    L2.Happiness Score                0.003541         0.002341            1.512           0.130
    L2.Playback Stalls               -0.049193         0.041017           -1.199           0.230
    L2.Startup Error (Count)          0.575178         0.197669            2.910           0.004
    L2.Latency                        0.000000         0.000000            0.712           0.476
    L2.Crash Status                  -0.327069         0.121237           -2.698           0.007
    L2.End of Playback Status         0.086286         0.052216            1.652           0.098
    L2.User_ID_N                      0.000182         0.000062            2.952           0.003
    L2.Title_N                        0.000019         0.000016            1.146           0.252
    L2.Device_Vendor_N               -0.000322         0.003457           -0.093           0.926
    L2.Device_Model_N                 0.001069         0.000407            2.625           0.009
    L2.Content_TV_Show_N              0.000009         0.000016            0.575           0.566
    L2.Country_N                      0.118664         0.005272           22.508           0.000
    L2.City_N                        -0.000189         0.000099           -1.919           0.055
    L2.Region_N                      -0.324001         0.098464           -3.291           0.001
    L3.Playtime                      -0.000004         0.000005           -0.911           0.362
    L3.Interruptions                 -0.000006         0.000623           -0.010           0.992
    L3.Join Time                     -0.000160         0.002844           -0.056           0.955
    L3.Buffer Ratio                   0.004045         0.002828            1.430           0.153
    L3.Connection Type                0.002246         0.002484            0.904           0.366
    L3.Device                         0.014322         0.007051            2.031           0.042
    L3.Device Type                   -0.011699         0.005756           -2.032           0.042
    L3.Browser                       -0.009227         0.008433           -1.094           0.274
    L3.OS                            -0.013018         0.026319           -0.495           0.621
    L3.OS Version                     0.001279         0.002751            0.465           0.642
    L3.Device ID                      0.000017         0.000019            0.880           0.379
    L3.Happiness Score                0.005700         0.002348            2.427           0.015
    L3.Playback Stalls                0.050716         0.041019            1.236           0.216
    L3.Startup Error (Count)          0.077438         0.197672            0.392           0.695
    L3.Latency                       -0.000000         0.000000           -0.449           0.654
    L3.Crash Status                   0.008408         0.121201            0.069           0.945
    L3.End of Playback Status         0.051501         0.052270            0.985           0.324
    L3.User_ID_N                      0.000003         0.000062            0.049           0.961
    L3.Title_N                        0.000025         0.000016            1.527           0.127
    L3.Device_Vendor_N               -0.001913         0.003481           -0.550           0.583
    L3.Device_Model_N                -0.000450         0.000409           -1.099           0.272
    L3.Content_TV_Show_N              0.000021         0.000016            1.257           0.209
    L3.Country_N                      0.096543         0.005302           18.208           0.000
    L3.City_N                         0.000005         0.000099            0.049           0.961
    L3.Region_N                      -0.453418         0.098866           -4.586           0.000
    L4.Playtime                      -0.000005         0.000005           -1.042           0.297
    L4.Interruptions                  0.000001         0.000623            0.002           0.998
    L4.Join Time                      0.000224         0.002829            0.079           0.937
    L4.Buffer Ratio                   0.004764         0.002828            1.685           0.092
    L4.Connection Type                0.003887         0.002494            1.559           0.119
    L4.Device                         0.002862         0.007067            0.405           0.685
    L4.Device Type                   -0.005799         0.005777           -1.004           0.315
    L4.Browser                        0.011769         0.008462            1.391           0.164
    L4.OS                            -0.010455         0.026390           -0.396           0.692
    L4.OS Version                    -0.000729         0.002758           -0.264           0.792
    L4.Device ID                      0.000051         0.000019            2.603           0.009
    L4.Happiness Score                0.001759         0.002350            0.748           0.454
    L4.Playback Stalls               -0.048938         0.041003           -1.194           0.233
    L4.Startup Error (Count)         -0.320105         0.197663           -1.619           0.105
    L4.Latency                       -0.000000         0.000000           -0.329           0.742
    L4.Crash Status                   0.246256         0.121201            2.032           0.042
    L4.End of Playback Status        -0.027674         0.052281           -0.529           0.597
    L4.User_ID_N                     -0.000050         0.000062           -0.799           0.424
    L4.Title_N                        0.000004         0.000016            0.238           0.812
    L4.Device_Vendor_N                0.006105         0.003493            1.748           0.081
    L4.Device_Model_N                -0.000169         0.000410           -0.412           0.680
    L4.Content_TV_Show_N              0.000033         0.000016            2.014           0.044
    L4.Country_N                      0.086186         0.005316           16.213           0.000
    L4.City_N                         0.000309         0.000100            3.098           0.002
    L4.Region_N                      -0.516772         0.099146           -5.212           0.000
    L5.Playtime                       0.000008         0.000005            1.598           0.110
    L5.Interruptions                 -0.000143         0.000623           -0.230           0.818
    L5.Join Time                     -0.001087         0.002830           -0.384           0.701
    L5.Buffer Ratio                   0.000604         0.002824            0.214           0.831
    L5.Connection Type               -0.002695         0.002496           -1.080           0.280
    L5.Device                         0.000462         0.007072            0.065           0.948
    L5.Device Type                    0.012143         0.005784            2.100           0.036
    L5.Browser                        0.006725         0.008475            0.793           0.428
    L5.OS                             0.028772         0.026417            1.089           0.276
    L5.OS Version                    -0.003141         0.002761           -1.137           0.255
    L5.Device ID                      0.000008         0.000020            0.386           0.699
    L5.Happiness Score               -0.000526         0.002351           -0.224           0.823
    L5.Playback Stalls                0.079546         0.039926            1.992           0.046
    L5.Startup Error (Count)          0.209258         0.197644            1.059           0.290
    L5.Latency                       -0.000001         0.000000           -1.853           0.064
    L5.Crash Status                  -0.119323         0.121217           -0.984           0.325
    L5.End of Playback Status         0.047509         0.052302            0.908           0.364
    L5.User_ID_N                     -0.000030         0.000062           -0.478           0.633
    L5.Title_N                        0.000002         0.000016            0.149           0.881
    L5.Device_Vendor_N               -0.000238         0.003496           -0.068           0.946
    L5.Device_Model_N                 0.000182         0.000411            0.443           0.658
    L5.Content_TV_Show_N              0.000004         0.000016            0.239           0.811
    L5.Country_N                      0.060054         0.005326           11.276           0.000
    L5.City_N                         0.000039         0.000100            0.395           0.693
    L5.Region_N                      -0.446573         0.099281           -4.498           0.000
    L6.Playtime                      -0.000003         0.000005           -0.579           0.563
    L6.Interruptions                 -0.000120         0.000623           -0.193           0.847
    L6.Join Time                     -0.000242         0.002831           -0.086           0.932
    L6.Buffer Ratio                  -0.002447         0.002824           -0.866           0.386
    L6.Connection Type                0.001912         0.002494            0.767           0.443
    L6.Device                        -0.005489         0.007068           -0.777           0.437
    L6.Device Type                   -0.006381         0.005778           -1.104           0.269
    L6.Browser                       -0.031395         0.008462           -3.710           0.000
    L6.OS                            -0.013640         0.026390           -0.517           0.605
    L6.OS Version                     0.008356         0.002759            3.029           0.002
    L6.Device ID                      0.000017         0.000019            0.885           0.376
    L6.Happiness Score               -0.002083         0.002349           -0.887           0.375
    L6.Playback Stalls                0.054716         0.040915            1.337           0.181
    L6.Startup Error (Count)          0.257229         0.197616            1.302           0.193
    L6.Latency                       -0.000000         0.000000           -1.176           0.240
    L6.Crash Status                  -0.077694         0.121208           -0.641           0.522
    L6.End of Playback Status         0.001501         0.052309            0.029           0.977
    L6.User_ID_N                      0.000086         0.000062            1.384           0.166
    L6.Title_N                       -0.000014         0.000016           -0.854           0.393
    L6.Device_Vendor_N               -0.003045         0.003493           -0.872           0.383
    L6.Device_Model_N                -0.000163         0.000410           -0.398           0.691
    L6.Content_TV_Show_N             -0.000002         0.000016           -0.135           0.893
    L6.Country_N                      0.061088         0.005315           11.494           0.000
    L6.City_N                         0.000242         0.000100            2.432           0.015
    L6.Region_N                      -0.655199         0.099133           -6.609           0.000
    L7.Playtime                       0.000001         0.000005            0.295           0.768
    L7.Interruptions                 -0.000127         0.000623           -0.203           0.839
    L7.Join Time                      0.002672         0.002830            0.944           0.345
    L7.Buffer Ratio                  -0.006442         0.002825           -2.281           0.023
    L7.Connection Type                0.003851         0.002484            1.551           0.121
    L7.Device                         0.010094         0.007049            1.432           0.152
    L7.Device Type                   -0.011422         0.005756           -1.984           0.047
    L7.Browser                        0.016763         0.008435            1.987           0.047
    L7.OS                            -0.019693         0.026317           -0.748           0.454
    L7.OS Version                    -0.001871         0.002752           -0.680           0.497
    L7.Device ID                      0.000013         0.000019            0.646           0.518
    L7.Happiness Score                0.000457         0.002346            0.195           0.846
    L7.Playback Stalls                0.018817         0.041128            0.458           0.647
    L7.Startup Error (Count)         -0.021178         0.197600           -0.107           0.915
    L7.Latency                       -0.000000         0.000000           -0.119           0.905
    L7.Crash Status                  -0.097644         0.121206           -0.806           0.420
    L7.End of Playback Status        -0.124160         0.052302           -2.374           0.018
    L7.User_ID_N                     -0.000030         0.000062           -0.489           0.625
    L7.Title_N                        0.000005         0.000016            0.289           0.773
    L7.Device_Vendor_N                0.002984         0.003482            0.857           0.391
    L7.Device_Model_N                 0.000473         0.000409            1.157           0.247
    L7.Content_TV_Show_N              0.000020         0.000016            1.208           0.227
    L7.Country_N                      0.041765         0.005301            7.879           0.000
    L7.City_N                         0.000041         0.000099            0.410           0.682
    L7.Region_N                      -0.368692         0.098861           -3.729           0.000
    L8.Playtime                       0.000003         0.000005            0.580           0.562
    L8.Interruptions                 -0.000091         0.000623           -0.146           0.884
    L8.Join Time                     -0.003105         0.002830           -1.097           0.273
    L8.Buffer Ratio                  -0.002723         0.002824           -0.964           0.335
    L8.Connection Type                0.001488         0.002464            0.604           0.546
    L8.Device                         0.019592         0.007013            2.794           0.005
    L8.Device Type                   -0.007303         0.005713           -1.278           0.201
    L8.Browser                       -0.015048         0.008390           -1.794           0.073
    L8.OS                            -0.018150         0.026219           -0.692           0.489
    L8.OS Version                     0.003365         0.002744            1.227           0.220
    L8.Device ID                      0.000053         0.000019            2.747           0.006
    L8.Happiness Score                0.002974         0.002340            1.271           0.204
    L8.Playback Stalls                0.083779         0.041129            2.037           0.042
    L8.Startup Error (Count)          0.200519         0.197459            1.015           0.310
    L8.Latency                       -0.000000         0.000000           -1.333           0.183
    L8.Crash Status                  -0.021475         0.121229           -0.177           0.859
    L8.End of Playback Status         0.068364         0.052289            1.307           0.191
    L8.User_ID_N                      0.000040         0.000062            0.657           0.511
    L8.Title_N                        0.000003         0.000016            0.172           0.863
    L8.Device_Vendor_N                0.000036         0.003457            0.010           0.992
    L8.Device_Model_N                -0.000861         0.000407           -2.116           0.034
    L8.Content_TV_Show_N              0.000015         0.000016            0.919           0.358
    L8.Country_N                      0.044644         0.005271            8.471           0.000
    L8.City_N                         0.000097         0.000099            0.986           0.324
    L8.Region_N                      -0.556760         0.098453           -5.655           0.000
    L9.Playtime                      -0.000006         0.000005           -1.239           0.215
    L9.Interruptions                  0.000005         0.000623            0.008           0.994
    L9.Join Time                      0.000937         0.002829            0.331           0.741
    L9.Buffer Ratio                   0.000442         0.002824            0.157           0.875
    L9.Connection Type               -0.002140         0.002433           -0.879           0.379
    L9.Device                        -0.004651         0.006943           -0.670           0.503
    L9.Device Type                    0.004466         0.005633            0.793           0.428
    L9.Browser                        0.004414         0.008269            0.534           0.593
    L9.OS                             0.040554         0.025913            1.565           0.118
    L9.OS Version                    -0.004256         0.002724           -1.562           0.118
    L9.Device ID                      0.000018         0.000019            0.958           0.338
    L9.Happiness Score               -0.002169         0.002308           -0.940           0.347
    L9.Playback Stalls                0.027175         0.040986            0.663           0.507
    L9.Startup Error (Count)          0.218162         0.197281            1.106           0.269
    L9.Latency                       -0.000000         0.000000           -0.644           0.520
    L9.Crash Status                  -0.096515         0.121128           -0.797           0.426
    L9.End of Playback Status         0.106115         0.052271            2.030           0.042
    L9.User_ID_N                      0.000017         0.000061            0.277           0.782
    L9.Title_N                       -0.000016         0.000016           -0.986           0.324
    L9.Device_Vendor_N                0.005017         0.003413            1.470           0.142
    L9.Device_Model_N                 0.000396         0.000402            0.985           0.324
    L9.Content_TV_Show_N             -0.000004         0.000016           -0.242           0.809
    L9.Country_N                      0.053002         0.005234           10.127           0.000
    L9.City_N                         0.000031         0.000097            0.316           0.752
    L9.Region_N                      -0.260863         0.097823           -2.667           0.008
    ============================================================================================
    
    Results for equation City_N
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             6.212284        16.285736            0.381           0.703
    L1.Playtime                      -0.000362         0.000199           -1.824           0.068
    L1.Interruptions                 -0.007156         0.025482           -0.281           0.779
    L1.Join Time                     -0.119133         0.116346           -1.024           0.306
    L1.Buffer Ratio                   0.237716         0.115424            2.060           0.039
    L1.Connection Type               -0.026372         0.099570           -0.265           0.791
    L1.Device                         0.189334         0.284320            0.666           0.505
    L1.Device Type                    0.782114         0.230652            3.391           0.001
    L1.Browser                       -0.701984         0.338356           -2.075           0.038
    L1.OS                             0.407722         1.061656            0.384           0.701
    L1.OS Version                     0.005842         0.111529            0.052           0.958
    L1.Device ID                      0.001558         0.000780            1.997           0.046
    L1.Happiness Score               -0.115482         0.094558           -1.221           0.222
    L1.Playback Stalls                0.321948         1.678475            0.192           0.848
    L1.Startup Error (Count)         18.404735         8.084867            2.276           0.023
    L1.Latency                       -0.000001         0.000015           -0.071           0.943
    L1.Crash Status                  -4.179135         4.949251           -0.844           0.398
    L1.End of Playback Status         5.645423         2.132232            2.648           0.008
    L1.User_ID_N                      0.000632         0.002480            0.255           0.799
    L1.Title_N                       -0.000290         0.000673           -0.431           0.666
    L1.Device_Vendor_N                0.331471         0.139660            2.373           0.018
    L1.Device_Model_N                 0.015857         0.016467            0.963           0.336
    L1.Content_TV_Show_N              0.000777         0.000667            1.166           0.244
    L1.Country_N                     -0.249218         0.214223           -1.163           0.245
    L1.City_N                         0.177733         0.003976           44.702           0.000
    L1.Region_N                      -0.703811         4.002272           -0.176           0.860
    L2.Playtime                       0.000005         0.000199            0.025           0.980
    L2.Interruptions                 -0.011768         0.025477           -0.462           0.644
    L2.Join Time                      0.070706         0.116379            0.608           0.543
    L2.Buffer Ratio                   0.179621         0.115699            1.552           0.121
    L2.Connection Type               -0.372361         0.100832           -3.693           0.000
    L2.Device                         0.127090         0.287085            0.443           0.658
    L2.Device Type                    0.062720         0.233781            0.268           0.788
    L2.Browser                        0.136120         0.343206            0.397           0.692
    L2.OS                             0.860524         1.073237            0.802           0.423
    L2.OS Version                    -0.248988         0.112302           -2.217           0.027
    L2.Device ID                      0.000389         0.000789            0.493           0.622
    L2.Happiness Score                0.059526         0.095804            0.621           0.534
    L2.Playback Stalls               -1.981838         1.678590           -1.181           0.238
    L2.Startup Error (Count)         10.131364         8.089511            1.252           0.210
    L2.Latency                       -0.000026         0.000015           -1.786           0.074
    L2.Crash Status                  -6.340755         4.961557           -1.278           0.201
    L2.End of Playback Status         1.856389         2.136917            0.869           0.385
    L2.User_ID_N                     -0.004548         0.002518           -1.806           0.071
    L2.Title_N                        0.000465         0.000674            0.690           0.490
    L2.Device_Vendor_N               -0.291962         0.141475           -2.064           0.039
    L2.Device_Model_N                -0.021999         0.016663           -1.320           0.187
    L2.Content_TV_Show_N              0.000816         0.000669            1.220           0.223
    L2.Country_N                     -0.527219         0.215760           -2.444           0.015
    L2.City_N                         0.131609         0.004037           32.603           0.000
    L2.Region_N                       1.490931         4.029584            0.370           0.711
    L3.Playtime                       0.000197         0.000199            0.993           0.321
    L3.Interruptions                 -0.007794         0.025476           -0.306           0.760
    L3.Join Time                      0.099199         0.116375            0.852           0.394
    L3.Buffer Ratio                   0.077804         0.115744            0.672           0.501
    L3.Connection Type               -0.135488         0.101651           -1.333           0.183
    L3.Device                         0.345175         0.288553            1.196           0.232
    L3.Device Type                   -0.434119         0.235574           -1.843           0.065
    L3.Browser                        0.183668         0.345135            0.532           0.595
    L3.OS                            -2.336462         1.077072           -2.169           0.030
    L3.OS Version                     0.145612         0.112597            1.293           0.196
    L3.Device ID                     -0.000826         0.000795           -1.039           0.299
    L3.Happiness Score                0.006445         0.096096            0.067           0.947
    L3.Playback Stalls               -0.608631         1.678683           -0.363           0.717
    L3.Startup Error (Count)         -8.325343         8.089649           -1.029           0.303
    L3.Latency                       -0.000016         0.000015           -1.110           0.267
    L3.Crash Status                   1.729232         4.960105            0.349           0.727
    L3.End of Playback Status        -0.455985         2.139129           -0.213           0.831
    L3.User_ID_N                     -0.001282         0.002537           -0.505           0.613
    L3.Title_N                        0.000284         0.000674            0.422           0.673
    L3.Device_Vendor_N                0.028417         0.142470            0.199           0.842
    L3.Device_Model_N                -0.002170         0.016750           -0.130           0.897
    L3.Content_TV_Show_N              0.002287         0.000671            3.409           0.001
    L3.Country_N                      0.420364         0.216997            1.937           0.053
    L3.City_N                         0.095151         0.004067           23.394           0.000
    L3.Region_N                     -10.661625         4.046061           -2.635           0.008
    L4.Playtime                       0.000306         0.000199            1.539           0.124
    L4.Interruptions                 -0.021581         0.025476           -0.847           0.397
    L4.Join Time                     -0.040863         0.115755           -0.353           0.724
    L4.Buffer Ratio                  -0.010913         0.115736           -0.094           0.925
    L4.Connection Type               -0.063799         0.102054           -0.625           0.532
    L4.Device                         0.618615         0.289221            2.139           0.032
    L4.Device Type                   -0.175062         0.236412           -0.740           0.459
    L4.Browser                        0.257068         0.346315            0.742           0.458
    L4.OS                            -1.548406         1.080005           -1.434           0.152
    L4.OS Version                     0.142581         0.112887            1.263           0.207
    L4.Device ID                     -0.000240         0.000798           -0.300           0.764
    L4.Happiness Score                0.197436         0.096192            2.053           0.040
    L4.Playback Stalls                0.053235         1.678009            0.032           0.975
    L4.Startup Error (Count)          6.496010         8.089249            0.803           0.422
    L4.Latency                        0.000002         0.000015            0.111           0.912
    L4.Crash Status                  -0.421116         4.960082           -0.085           0.932
    L4.End of Playback Status         3.855274         2.139589            1.802           0.072
    L4.User_ID_N                     -0.003411         0.002545           -1.340           0.180
    L4.Title_N                        0.000961         0.000674            1.426           0.154
    L4.Device_Vendor_N                0.295314         0.142961            2.066           0.039
    L4.Device_Model_N                 0.003650         0.016792            0.217           0.828
    L4.Content_TV_Show_N             -0.000073         0.000672           -0.109           0.913
    L4.Country_N                      0.037493         0.217546            0.172           0.863
    L4.City_N                         0.074411         0.004080           18.238           0.000
    L4.Region_N                     -10.643092         4.057520           -2.623           0.009
    L5.Playtime                       0.000063         0.000199            0.318           0.750
    L5.Interruptions                 -0.007684         0.025477           -0.302           0.763
    L5.Join Time                     -0.047406         0.115800           -0.409           0.682
    L5.Buffer Ratio                  -0.060268         0.115568           -0.521           0.602
    L5.Connection Type               -0.103632         0.102140           -1.015           0.310
    L5.Device                        -0.295869         0.289424           -1.022           0.307
    L5.Device Type                   -0.119226         0.236701           -0.504           0.614
    L5.Browser                        0.548214         0.346854            1.581           0.114
    L5.OS                             0.629452         1.081083            0.582           0.560
    L5.OS Version                    -0.136881         0.113002           -1.211           0.226
    L5.Device ID                     -0.000323         0.000798           -0.405           0.686
    L5.Happiness Score               -0.015654         0.096205           -0.163           0.871
    L5.Playback Stalls               -0.774349         1.633972           -0.474           0.636
    L5.Startup Error (Count)         13.004396         8.088489            1.608           0.108
    L5.Latency                        0.000014         0.000015            0.957           0.339
    L5.Crash Status                 -10.769436         4.960754           -2.171           0.030
    L5.End of Playback Status         4.794796         2.140430            2.240           0.025
    L5.User_ID_N                      0.002355         0.002547            0.924           0.355
    L5.Title_N                        0.000517         0.000674            0.768           0.443
    L5.Device_Vendor_N                0.107211         0.143064            0.749           0.454
    L5.Device_Model_N                 0.023388         0.016803            1.392           0.164
    L5.Content_TV_Show_N             -0.001233         0.000672           -1.834           0.067
    L5.Country_N                      0.149220         0.217951            0.685           0.494
    L5.City_N                         0.051436         0.004086           12.590           0.000
    L5.Region_N                      -0.123842         4.063025           -0.030           0.976
    L6.Playtime                       0.000017         0.000199            0.088           0.930
    L6.Interruptions                 -0.002319         0.025477           -0.091           0.927
    L6.Join Time                      0.036871         0.115846            0.318           0.750
    L6.Buffer Ratio                  -0.094306         0.115573           -0.816           0.415
    L6.Connection Type               -0.088502         0.102055           -0.867           0.386
    L6.Device                         0.153684         0.289247            0.531           0.595
    L6.Device Type                    0.416634         0.236479            1.762           0.078
    L6.Browser                       -0.173563         0.346298           -0.501           0.616
    L6.OS                            -0.146725         1.080011           -0.136           0.892
    L6.OS Version                    -0.063336         0.112896           -0.561           0.575
    L6.Device ID                      0.000910         0.000798            1.141           0.254
    L6.Happiness Score               -0.238392         0.096135           -2.480           0.013
    L6.Playback Stalls                3.489510         1.674427            2.084           0.037
    L6.Startup Error (Count)         -8.029396         8.087358           -0.993           0.321
    L6.Latency                        0.000003         0.000015            0.203           0.839
    L6.Crash Status                   7.466557         4.960397            1.505           0.132
    L6.End of Playback Status        -2.239908         2.140741           -1.046           0.295
    L6.User_ID_N                      0.002350         0.002545            0.923           0.356
    L6.Title_N                        0.000089         0.000674            0.132           0.895
    L6.Device_Vendor_N               -0.346635         0.142946           -2.425           0.015
    L6.Device_Model_N                -0.028449         0.016789           -1.694           0.090
    L6.Content_TV_Show_N             -0.001179         0.000672           -1.755           0.079
    L6.Country_N                      0.151588         0.217510            0.697           0.486
    L6.City_N                         0.048636         0.004080           11.920           0.000
    L6.Region_N                      -5.859718         4.056977           -1.444           0.149
    L7.Playtime                      -0.000154         0.000199           -0.775           0.439
    L7.Interruptions                 -0.001545         0.025478           -0.061           0.952
    L7.Join Time                      0.116895         0.115832            1.009           0.313
    L7.Buffer Ratio                  -0.171184         0.115593           -1.481           0.139
    L7.Connection Type                0.163156         0.101649            1.605           0.108
    L7.Device                         0.673624         0.288466            2.335           0.020
    L7.Device Type                   -0.492149         0.235581           -2.089           0.037
    L7.Browser                        0.728780         0.345189            2.111           0.035
    L7.OS                            -1.619585         1.076995           -1.504           0.133
    L7.OS Version                     0.009423         0.112617            0.084           0.933
    L7.Device ID                      0.001037         0.000795            1.305           0.192
    L7.Happiness Score                0.087321         0.096005            0.910           0.363
    L7.Playback Stalls                3.780494         1.683132            2.246           0.025
    L7.Startup Error (Count)          9.304925         8.086672            1.151           0.250
    L7.Latency                       -0.000002         0.000015           -0.139           0.889
    L7.Crash Status                  -5.665694         4.960307           -1.142           0.253
    L7.End of Playback Status         3.574372         2.140446            1.670           0.095
    L7.User_ID_N                      0.002403         0.002537            0.947           0.344
    L7.Title_N                        0.001708         0.000674            2.534           0.011
    L7.Device_Vendor_N                0.212164         0.142489            1.489           0.136
    L7.Device_Model_N                 0.002589         0.016747            0.155           0.877
    L7.Content_TV_Show_N              0.000267         0.000671            0.398           0.691
    L7.Country_N                     -0.439296         0.216926           -2.025           0.043
    L7.City_N                         0.043321         0.004067           10.651           0.000
    L7.Region_N                      -2.853568         4.045835           -0.705           0.481
    L8.Playtime                       0.000270         0.000199            1.358           0.174
    L8.Interruptions                 -0.019256         0.025477           -0.756           0.450
    L8.Join Time                      0.161475         0.115820            1.394           0.163
    L8.Buffer Ratio                  -0.237446         0.115574           -2.054           0.040
    L8.Connection Type               -0.076128         0.100841           -0.755           0.450
    L8.Device                        -0.042986         0.286994           -0.150           0.881
    L8.Device Type                   -0.189105         0.233807           -0.809           0.419
    L8.Browser                       -0.623819         0.343355           -1.817           0.069
    L8.OS                            -1.575416         1.073019           -1.468           0.142
    L8.OS Version                     0.293815         0.112285            2.617           0.009
    L8.Device ID                      0.000983         0.000789            1.245           0.213
    L8.Happiness Score                0.059206         0.095747            0.618           0.536
    L8.Playback Stalls               -0.210993         1.683167           -0.125           0.900
    L8.Startup Error (Count)         -3.220863         8.080919           -0.399           0.690
    L8.Latency                       -0.000015         0.000015           -1.031           0.303
    L8.Crash Status                   1.387111         4.961245            0.280           0.780
    L8.End of Playback Status        -1.901877         2.139921           -0.889           0.374
    L8.User_ID_N                     -0.000553         0.002518           -0.220           0.826
    L8.Title_N                       -0.000281         0.000673           -0.417           0.676
    L8.Device_Vendor_N               -0.004543         0.141485           -0.032           0.974
    L8.Device_Model_N                 0.014885         0.016661            0.893           0.372
    L8.Content_TV_Show_N              0.000788         0.000669            1.178           0.239
    L8.Country_N                      0.055101         0.215694            0.255           0.798
    L8.City_N                         0.045149         0.004036           11.186           0.000
    L8.Region_N                      -7.293113         4.029153           -1.810           0.070
    L9.Playtime                      -0.000114         0.000199           -0.572           0.567
    L9.Interruptions                 -0.000429         0.025477           -0.017           0.987
    L9.Join Time                      0.039154         0.115755            0.338           0.735
    L9.Buffer Ratio                  -0.187991         0.115551           -1.627           0.104
    L9.Connection Type               -0.029289         0.099580           -0.294           0.769
    L9.Device                        -0.000268         0.284138           -0.001           0.999
    L9.Device Type                   -0.233156         0.230526           -1.011           0.312
    L9.Browser                        0.285830         0.338412            0.845           0.398
    L9.OS                            -0.311714         1.060496           -0.294           0.769
    L9.OS Version                    -0.022643         0.111491           -0.203           0.839
    L9.Device ID                      0.001102         0.000781            1.412           0.158
    L9.Happiness Score                0.057592         0.094448            0.610           0.542
    L9.Playback Stalls               -0.917533         1.677333           -0.547           0.584
    L9.Startup Error (Count)         -5.788142         8.073637           -0.717           0.473
    L9.Latency                       -0.000008         0.000015           -0.534           0.593
    L9.Crash Status                   7.097998         4.957111            1.432           0.152
    L9.End of Playback Status        -0.244778         2.139151           -0.114           0.909
    L9.User_ID_N                     -0.000650         0.002481           -0.262           0.793
    L9.Title_N                       -0.000300         0.000672           -0.446           0.655
    L9.Device_Vendor_N                0.067292         0.139661            0.482           0.630
    L9.Device_Model_N                -0.003737         0.016466           -0.227           0.820
    L9.Content_TV_Show_N             -0.001080         0.000666           -1.620           0.105
    L9.Country_N                      0.265634         0.214189            1.240           0.215
    L9.City_N                         0.047913         0.003979           12.041           0.000
    L9.Region_N                      -2.725362         4.003352           -0.681           0.496
    ============================================================================================
    
    Results for equation Region_N
    ============================================================================================
                                   coefficient       std. error           t-stat            prob
    --------------------------------------------------------------------------------------------
    const                             0.004057         0.025008            0.162           0.871
    L1.Playtime                       0.000000         0.000000            0.772           0.440
    L1.Interruptions                 -0.000013         0.000039           -0.332           0.740
    L1.Join Time                      0.000271         0.000179            1.519           0.129
    L1.Buffer Ratio                   0.000184         0.000177            1.041           0.298
    L1.Connection Type                0.000046         0.000153            0.303           0.762
    L1.Device                         0.000372         0.000437            0.851           0.395
    L1.Device Type                   -0.000383         0.000354           -1.082           0.279
    L1.Browser                       -0.000788         0.000520           -1.517           0.129
    L1.OS                            -0.008225         0.001630           -5.045           0.000
    L1.OS Version                     0.001147         0.000171            6.698           0.000
    L1.Device ID                     -0.000001         0.000001           -1.101           0.271
    L1.Happiness Score                0.000108         0.000145            0.746           0.455
    L1.Playback Stalls               -0.001057         0.002577           -0.410           0.682
    L1.Startup Error (Count)         -0.004563         0.012415           -0.368           0.713
    L1.Latency                        0.000000         0.000000            1.299           0.194
    L1.Crash Status                  -0.001027         0.007600           -0.135           0.892
    L1.End of Playback Status         0.000727         0.003274            0.222           0.824
    L1.User_ID_N                     -0.000004         0.000004           -1.155           0.248
    L1.Title_N                       -0.000002         0.000001           -1.841           0.066
    L1.Device_Vendor_N                0.000081         0.000214            0.378           0.706
    L1.Device_Model_N                 0.000066         0.000025            2.624           0.009
    L1.Content_TV_Show_N             -0.000003         0.000001           -2.697           0.007
    L1.Country_N                     -0.000450         0.000329           -1.367           0.172
    L1.City_N                        -0.000006         0.000006           -0.975           0.330
    L1.Region_N                       0.139126         0.006146           22.638           0.000
    L2.Playtime                       0.000000         0.000000            1.374           0.169
    L2.Interruptions                 -0.000010         0.000039           -0.265           0.791
    L2.Join Time                     -0.000028         0.000179           -0.159           0.874
    L2.Buffer Ratio                   0.000283         0.000178            1.593           0.111
    L2.Connection Type               -0.000029         0.000155           -0.185           0.853
    L2.Device                         0.000201         0.000441            0.457           0.648
    L2.Device Type                   -0.000486         0.000359           -1.355           0.175
    L2.Browser                       -0.000338         0.000527           -0.642           0.521
    L2.OS                            -0.002089         0.001648           -1.268           0.205
    L2.OS Version                     0.000217         0.000172            1.259           0.208
    L2.Device ID                      0.000001         0.000001            0.663           0.508
    L2.Happiness Score                0.000380         0.000147            2.585           0.010
    L2.Playback Stalls               -0.000260         0.002578           -0.101           0.920
    L2.Startup Error (Count)          0.026165         0.012422            2.106           0.035
    L2.Latency                        0.000000         0.000000            0.522           0.601
    L2.Crash Status                  -0.016859         0.007619           -2.213           0.027
    L2.End of Playback Status         0.000210         0.003281            0.064           0.949
    L2.User_ID_N                      0.000003         0.000004            0.842           0.400
    L2.Title_N                        0.000002         0.000001            2.408           0.016
    L2.Device_Vendor_N               -0.000123         0.000217           -0.568           0.570
    L2.Device_Model_N                 0.000012         0.000026            0.470           0.638
    L2.Content_TV_Show_N             -0.000000         0.000001           -0.184           0.854
    L2.Country_N                      0.000003         0.000331            0.009           0.993
    L2.City_N                        -0.000003         0.000006           -0.409           0.682
    L2.Region_N                       0.095010         0.006188           15.355           0.000
    L3.Playtime                       0.000000         0.000000            0.387           0.698
    L3.Interruptions                 -0.000009         0.000039           -0.234           0.815
    L3.Join Time                      0.000019         0.000179            0.109           0.913
    L3.Buffer Ratio                  -0.000171         0.000178           -0.960           0.337
    L3.Connection Type               -0.000124         0.000156           -0.792           0.429
    L3.Device                         0.000529         0.000443            1.194           0.233
    L3.Device Type                   -0.000025         0.000362           -0.068           0.946
    L3.Browser                       -0.000122         0.000530           -0.231           0.817
    L3.OS                            -0.001509         0.001654           -0.912           0.362
    L3.OS Version                     0.000168         0.000173            0.972           0.331
    L3.Device ID                      0.000000         0.000001            0.359           0.719
    L3.Happiness Score                0.000415         0.000148            2.809           0.005
    L3.Playback Stalls                0.000213         0.002578            0.083           0.934
    L3.Startup Error (Count)          0.003940         0.012422            0.317           0.751
    L3.Latency                       -0.000000         0.000000           -0.581           0.561
    L3.Crash Status                  -0.000008         0.007617           -0.001           0.999
    L3.End of Playback Status         0.001959         0.003285            0.596           0.551
    L3.User_ID_N                     -0.000007         0.000004           -1.913           0.056
    L3.Title_N                        0.000001         0.000001            1.331           0.183
    L3.Device_Vendor_N                0.000036         0.000219            0.164           0.870
    L3.Device_Model_N                 0.000012         0.000026            0.482           0.629
    L3.Content_TV_Show_N              0.000002         0.000001            1.773           0.076
    L3.Country_N                     -0.000150         0.000333           -0.451           0.652
    L3.City_N                        -0.000001         0.000006           -0.148           0.882
    L3.Region_N                       0.059486         0.006213            9.575           0.000
    L4.Playtime                      -0.000000         0.000000           -0.436           0.663
    L4.Interruptions                 -0.000000         0.000039           -0.004           0.997
    L4.Join Time                     -0.000097         0.000178           -0.548           0.583
    L4.Buffer Ratio                   0.000132         0.000178            0.744           0.457
    L4.Connection Type                0.000228         0.000157            1.453           0.146
    L4.Device                        -0.000353         0.000444           -0.796           0.426
    L4.Device Type                    0.000018         0.000363            0.050           0.960
    L4.Browser                        0.001043         0.000532            1.961           0.050
    L4.OS                             0.001171         0.001658            0.706           0.480
    L4.OS Version                    -0.000246         0.000173           -1.418           0.156
    L4.Device ID                      0.000001         0.000001            0.587           0.557
    L4.Happiness Score               -0.000004         0.000148           -0.026           0.979
    L4.Playback Stalls               -0.000464         0.002577           -0.180           0.857
    L4.Startup Error (Count)         -0.001548         0.012422           -0.125           0.901
    L4.Latency                       -0.000000         0.000000           -0.259           0.796
    L4.Crash Status                   0.006220         0.007617            0.817           0.414
    L4.End of Playback Status         0.003562         0.003285            1.084           0.278
    L4.User_ID_N                      0.000001         0.000004            0.316           0.752
    L4.Title_N                       -0.000000         0.000001           -0.021           0.983
    L4.Device_Vendor_N                0.000090         0.000220            0.411           0.681
    L4.Device_Model_N                -0.000027         0.000026           -1.031           0.303
    L4.Content_TV_Show_N              0.000000         0.000001            0.462           0.644
    L4.Country_N                     -0.000287         0.000334           -0.859           0.391
    L4.City_N                         0.000013         0.000006            2.030           0.042
    L4.Region_N                       0.043297         0.006231            6.949           0.000
    L5.Playtime                       0.000001         0.000000            2.056           0.040
    L5.Interruptions                 -0.000016         0.000039           -0.410           0.682
    L5.Join Time                     -0.000090         0.000178           -0.505           0.614
    L5.Buffer Ratio                   0.000066         0.000177            0.371           0.710
    L5.Connection Type               -0.000226         0.000157           -1.439           0.150
    L5.Device                        -0.000692         0.000444           -1.557           0.119
    L5.Device Type                    0.000634         0.000363            1.745           0.081
    L5.Browser                        0.000382         0.000533            0.717           0.473
    L5.OS                             0.002305         0.001660            1.388           0.165
    L5.OS Version                    -0.000251         0.000174           -1.448           0.148
    L5.Device ID                     -0.000001         0.000001           -0.551           0.582
    L5.Happiness Score                0.000078         0.000148            0.529           0.597
    L5.Playback Stalls               -0.000944         0.002509           -0.376           0.707
    L5.Startup Error (Count)          0.014396         0.012420            1.159           0.246
    L5.Latency                       -0.000000         0.000000           -2.718           0.007
    L5.Crash Status                  -0.006374         0.007618           -0.837           0.403
    L5.End of Playback Status         0.004308         0.003287            1.311           0.190
    L5.User_ID_N                     -0.000003         0.000004           -0.720           0.472
    L5.Title_N                       -0.000000         0.000001           -0.446           0.656
    L5.Device_Vendor_N               -0.000107         0.000220           -0.487           0.626
    L5.Device_Model_N                 0.000001         0.000026            0.026           0.979
    L5.Content_TV_Show_N             -0.000003         0.000001           -2.450           0.014
    L5.Country_N                      0.000041         0.000335            0.122           0.903
    L5.City_N                         0.000002         0.000006            0.242           0.809
    L5.Region_N                       0.027803         0.006239            4.456           0.000
    L6.Playtime                       0.000001         0.000000            2.291           0.022
    L6.Interruptions                 -0.000022         0.000039           -0.575           0.565
    L6.Join Time                     -0.000179         0.000178           -1.005           0.315
    L6.Buffer Ratio                   0.000005         0.000177            0.026           0.979
    L6.Connection Type                0.000270         0.000157            1.723           0.085
    L6.Device                         0.000258         0.000444            0.580           0.562
    L6.Device Type                    0.000035         0.000363            0.095           0.924
    L6.Browser                       -0.000904         0.000532           -1.700           0.089
    L6.OS                            -0.002222         0.001658           -1.340           0.180
    L6.OS Version                     0.000451         0.000173            2.604           0.009
    L6.Device ID                      0.000000         0.000001            0.291           0.771
    L6.Happiness Score               -0.000108         0.000148           -0.731           0.465
    L6.Playback Stalls                0.000042         0.002571            0.016           0.987
    L6.Startup Error (Count)          0.003889         0.012419            0.313           0.754
    L6.Latency                       -0.000000         0.000000           -0.961           0.337
    L6.Crash Status                  -0.002379         0.007617           -0.312           0.755
    L6.End of Playback Status         0.000913         0.003287            0.278           0.781
    L6.User_ID_N                     -0.000002         0.000004           -0.637           0.524
    L6.Title_N                       -0.000001         0.000001           -0.630           0.529
    L6.Device_Vendor_N                0.000178         0.000220            0.812           0.417
    L6.Device_Model_N                 0.000011         0.000026            0.435           0.663
    L6.Content_TV_Show_N              0.000000         0.000001            0.155           0.877
    L6.Country_N                     -0.000195         0.000334           -0.583           0.560
    L6.City_N                         0.000013         0.000006            2.084           0.037
    L6.Region_N                       0.014679         0.006230            2.356           0.018
    L7.Playtime                       0.000001         0.000000            1.685           0.092
    L7.Interruptions                 -0.000014         0.000039           -0.355           0.723
    L7.Join Time                      0.000267         0.000178            1.501           0.133
    L7.Buffer Ratio                  -0.000247         0.000177           -1.393           0.164
    L7.Connection Type                0.000000         0.000156            0.002           0.998
    L7.Device                         0.000218         0.000443            0.492           0.623
    L7.Device Type                   -0.000476         0.000362           -1.317           0.188
    L7.Browser                        0.001115         0.000530            2.103           0.035
    L7.OS                            -0.001744         0.001654           -1.055           0.292
    L7.OS Version                    -0.000013         0.000173           -0.077           0.939
    L7.Device ID                     -0.000000         0.000001           -0.397           0.691
    L7.Happiness Score                0.000139         0.000147            0.944           0.345
    L7.Playback Stalls               -0.000357         0.002585           -0.138           0.890
    L7.Startup Error (Count)          0.018713         0.012418            1.507           0.132
    L7.Latency                        0.000000         0.000000            0.206           0.837
    L7.Crash Status                  -0.015760         0.007617           -2.069           0.039
    L7.End of Playback Status        -0.000274         0.003287           -0.083           0.934
    L7.User_ID_N                     -0.000005         0.000004           -1.319           0.187
    L7.Title_N                        0.000001         0.000001            0.581           0.562
    L7.Device_Vendor_N               -0.000072         0.000219           -0.327           0.744
    L7.Device_Model_N                 0.000032         0.000026            1.230           0.219
    L7.Content_TV_Show_N              0.000000         0.000001            0.418           0.676
    L7.Country_N                     -0.000670         0.000333           -2.012           0.044
    L7.City_N                         0.000003         0.000006            0.484           0.628
    L7.Region_N                       0.015226         0.006213            2.451           0.014
    L8.Playtime                       0.000001         0.000000            1.907           0.057
    L8.Interruptions                 -0.000019         0.000039           -0.477           0.633
    L8.Join Time                     -0.000008         0.000178           -0.044           0.965
    L8.Buffer Ratio                  -0.000131         0.000177           -0.740           0.459
    L8.Connection Type               -0.000005         0.000155           -0.033           0.974
    L8.Device                        -0.000072         0.000441           -0.164           0.869
    L8.Device Type                   -0.000004         0.000359           -0.012           0.991
    L8.Browser                       -0.000258         0.000527           -0.490           0.624
    L8.OS                            -0.000296         0.001648           -0.180           0.857
    L8.OS Version                     0.000077         0.000172            0.448           0.654
    L8.Device ID                      0.000002         0.000001            1.984           0.047
    L8.Happiness Score                0.000137         0.000147            0.935           0.350
    L8.Playback Stalls                0.000226         0.002585            0.087           0.930
    L8.Startup Error (Count)         -0.001841         0.012409           -0.148           0.882
    L8.Latency                       -0.000000         0.000000           -1.501           0.133
    L8.Crash Status                   0.003965         0.007618            0.520           0.603
    L8.End of Playback Status         0.000545         0.003286            0.166           0.868
    L8.User_ID_N                      0.000002         0.000004            0.579           0.563
    L8.Title_N                       -0.000001         0.000001           -0.886           0.376
    L8.Device_Vendor_N               -0.000143         0.000217           -0.657           0.511
    L8.Device_Model_N                -0.000005         0.000026           -0.193           0.847
    L8.Content_TV_Show_N              0.000002         0.000001            1.612           0.107
    L8.Country_N                     -0.000011         0.000331           -0.033           0.973
    L8.City_N                         0.000007         0.000006            1.074           0.283
    L8.Region_N                       0.012158         0.006187            1.965           0.049
    L9.Playtime                      -0.000000         0.000000           -0.387           0.699
    L9.Interruptions                  0.000002         0.000039            0.045           0.964
    L9.Join Time                      0.000181         0.000178            1.017           0.309
    L9.Buffer Ratio                   0.000110         0.000177            0.622           0.534
    L9.Connection Type               -0.000387         0.000153           -2.531           0.011
    L9.Device                        -0.001052         0.000436           -2.410           0.016
    L9.Device Type                    0.000379         0.000354            1.072           0.284
    L9.Browser                       -0.000320         0.000520           -0.615           0.538
    L9.OS                             0.002018         0.001628            1.239           0.215
    L9.OS Version                    -0.000095         0.000171           -0.553           0.580
    L9.Device ID                     -0.000000         0.000001           -0.273           0.785
    L9.Happiness Score               -0.000024         0.000145           -0.166           0.868
    L9.Playback Stalls               -0.000161         0.002576           -0.063           0.950
    L9.Startup Error (Count)         -0.011673         0.012398           -0.942           0.346
    L9.Latency                       -0.000000         0.000000           -0.147           0.883
    L9.Crash Status                   0.009101         0.007612            1.196           0.232
    L9.End of Playback Status         0.001165         0.003285            0.355           0.723
    L9.User_ID_N                     -0.000004         0.000004           -1.175           0.240
    L9.Title_N                       -0.000001         0.000001           -0.955           0.340
    L9.Device_Vendor_N               -0.000189         0.000214           -0.883           0.377
    L9.Device_Model_N                 0.000016         0.000025            0.628           0.530
    L9.Content_TV_Show_N              0.000000         0.000001            0.003           0.998
    L9.Country_N                     -0.000038         0.000329           -0.114           0.909
    L9.City_N                         0.000010         0.000006            1.704           0.088
    L9.Region_N                       0.028284         0.006147            4.601           0.000
    ============================================================================================
    
    Correlation matrix of residuals
                              Playtime  Interruptions  Join Time  Buffer Ratio  Connection Type    Device  Device Type   Browser        OS  OS Version  Device ID  Happiness Score  Playback Stalls  Startup Error (Count)   Latency  Crash Status  End of Playback Status  User_ID_N   Title_N  Device_Vendor_N  Device_Model_N  Content_TV_Show_N  Country_N    City_N  Region_N
    Playtime                  1.000000       0.179828   0.046284     -0.006663         0.012674  0.008300    -0.031330  0.008046  0.015967    0.020322   0.043887         0.113544        -0.001422              -0.009226  0.015575     -0.027904               -0.015040  -0.029306  0.050493         0.000186        0.001953          -0.023045   0.008267  0.039672  0.037371
    Interruptions             0.179828       1.000000   0.004231      0.015895         0.000378  0.011646    -0.012739  0.005305  0.014288    0.009472   0.001088        -0.001562         0.001425              -0.000756  0.004628     -0.001863               -0.003847   0.005850  0.007239        -0.006894        0.014695          -0.008083   0.000869  0.004462 -0.001537
    Join Time                 0.046284       0.004231   1.000000      0.032848         0.048156 -0.020818     0.009092  0.035868  0.008459    0.007158   0.020771         0.019853         0.030045              -0.005119  0.012743     -0.043002                0.017754   0.010002  0.003604         0.023804       -0.057812          -0.020789   0.006460 -0.011100 -0.063825
    Buffer Ratio             -0.006663       0.015895   0.032848      1.000000         0.014491  0.063898    -0.050862  0.083761  0.068743    0.054225  -0.003924        -0.039545         0.365216              -0.002472  0.044250     -0.045559               -0.294239   0.035242  0.003922        -0.066665        0.055364          -0.035455   0.061159  0.021106 -0.007473
    Connection Type           0.012674       0.000378   0.048156      0.014491         1.000000 -0.040363     0.198268  0.039963 -0.055961   -0.013128  -0.210650         0.004972         0.004120               0.002908  0.032135     -0.004708               -0.001106  -0.082098  0.067547        -0.119568        0.045935           0.022482   0.122065 -0.103554  0.051434
    Device                    0.008300       0.011646  -0.020818      0.063898        -0.040363  1.000000    -0.434431  0.713472  0.897340    0.821313  -0.090417        -0.302877         0.003084               0.011354 -0.116623     -0.004982               -0.049909   0.060910  0.081874        -0.754006        0.622935          -0.366904   0.515903  0.305708  0.525084
    Device Type              -0.031330      -0.012739   0.009092     -0.050862         0.198268 -0.434431     1.000000 -0.200697 -0.484926   -0.319341   0.004250         0.212898        -0.002052              -0.042191  0.085002      0.006585                0.104010   0.044834 -0.142466         0.415351       -0.371426           0.434617  -0.186765 -0.116663 -0.062811
    Browser                   0.008046       0.005305   0.035868      0.083761         0.039963  0.713472    -0.200697  1.000000  0.828807    0.851754  -0.037681        -0.248568         0.009296               0.025834 -0.143029      0.003277               -0.060590   0.030225  0.064071        -0.617490        0.340814          -0.343778   0.509615  0.274515  0.443472
    OS                        0.015967       0.014288   0.008459      0.068743        -0.055961  0.897340    -0.484926  0.828807  1.000000    0.935765  -0.032868        -0.291704         0.003274               0.029449 -0.122991      0.001145               -0.075568   0.025448  0.098206        -0.770791        0.580597          -0.420510   0.499450  0.282426  0.460081
    OS Version                0.020322       0.009472   0.007158      0.054225        -0.013128  0.821313    -0.319341  0.851754  0.935765    1.000000   0.001189        -0.292158         0.003182               0.033422 -0.137443      0.012759               -0.058433  -0.009717  0.091711        -0.735839        0.453008          -0.390534   0.578100  0.316091  0.612007
    Device ID                 0.043887       0.001088   0.020771     -0.003924        -0.210650 -0.090417     0.004250 -0.037681 -0.032868    0.001189   1.000000        -0.079756         0.005830               0.029682 -0.076914      0.010328               -0.039621   0.026050 -0.025783         0.091083        0.110205          -0.013960   0.057671  0.037109  0.048584
    Happiness Score           0.113544      -0.001562   0.019853     -0.039545         0.004972 -0.302877     0.212898 -0.248568 -0.291704   -0.292158  -0.079756         1.000000        -0.011749              -0.131900  0.252055     -0.039751                0.147637   0.091496 -0.189632         0.230192       -0.215791           0.151639  -0.260482 -0.247981 -0.216974
    Playback Stalls          -0.001422       0.001425   0.030045      0.365216         0.004120  0.003084    -0.002052  0.009296  0.003274    0.003182   0.005830        -0.011749         1.000000              -0.000720  0.015148     -0.062136               -0.094566   0.001055  0.002445        -0.003325       -0.001790          -0.000987   0.004101  0.002546 -0.001458
    Startup Error (Count)    -0.009226      -0.000756  -0.005119     -0.002472         0.002908  0.011354    -0.042191  0.025834  0.029449    0.033422   0.029682        -0.131900        -0.000720               1.000000 -0.034708      0.769384               -0.675314  -0.047969  0.022281        -0.027307        0.014437          -0.044056   0.028383  0.003467  0.015772
    Latency                   0.015575       0.004628   0.012743      0.044250         0.032135 -0.116623     0.085002 -0.143029 -0.122991   -0.137443  -0.076914         0.252055         0.015148              -0.034708  1.000000     -0.025058                0.013584  -0.002115 -0.072131         0.084647       -0.072836           0.061996  -0.115659 -0.103607 -0.093961
    Crash Status             -0.027904      -0.001863  -0.043002     -0.045559        -0.004708 -0.004982     0.006585  0.003277  0.001145    0.012759   0.010328        -0.039751        -0.062136               0.769384 -0.025058      1.000000               -0.257122  -0.037541 -0.000271        -0.000621       -0.032968          -0.025007  -0.000557  0.008023  0.020161
    End of Playback Status   -0.015040      -0.003847   0.017754     -0.294239        -0.001106 -0.049909     0.104010 -0.060590 -0.075568   -0.058433  -0.039621         0.147637        -0.094566              -0.675314  0.013584     -0.257122                1.000000   0.023137 -0.034904         0.071140       -0.087340           0.066333  -0.061622 -0.010280 -0.004928
    User_ID_N                -0.029306       0.005850   0.010002      0.035242        -0.082098  0.060910     0.044834  0.030225  0.025448   -0.009717   0.026050         0.091496         0.001055              -0.047969 -0.002115     -0.037541                0.023137   1.000000 -0.148435        -0.056942        0.086382           0.032815   0.034263 -0.157804 -0.088392
    Title_N                   0.050493       0.007239   0.003604      0.003922         0.067547  0.081874    -0.142466  0.064071  0.098206    0.091711  -0.025783        -0.189632         0.002445               0.022281 -0.072131     -0.000271               -0.034904  -0.148435  1.000000        -0.074457        0.070367          -0.051427   0.068496  0.185122  0.059788
    Device_Vendor_N           0.000186      -0.006894   0.023804     -0.066665        -0.119568 -0.754006     0.415351 -0.617490 -0.770791   -0.735839   0.091083         0.230192        -0.003325              -0.027307  0.084647     -0.000621                0.071140  -0.056942 -0.074457         1.000000       -0.556415           0.325692  -0.453869 -0.171950 -0.409303
    Device_Model_N            0.001953       0.014695  -0.057812      0.055364         0.045935  0.622935    -0.371426  0.340814  0.580597    0.453008   0.110205        -0.215791        -0.001790               0.014437 -0.072836     -0.032968               -0.087340   0.086382  0.070367        -0.556415        1.000000          -0.252196   0.370141  0.122258  0.273226
    Content_TV_Show_N        -0.023045      -0.008083  -0.020789     -0.035455         0.022482 -0.366904     0.434617 -0.343778 -0.420510   -0.390534  -0.013960         0.151639        -0.000987              -0.044056  0.061996     -0.025007                0.066333   0.032815 -0.051427         0.325692       -0.252196           1.000000  -0.277856 -0.150678 -0.193721
    Country_N                 0.008267       0.000869   0.006460      0.061159         0.122065  0.515903    -0.186765  0.509615  0.499450    0.578100   0.057671        -0.260482         0.004101               0.028383 -0.115659     -0.000557               -0.061622   0.034263  0.068496        -0.453869        0.370141          -0.277856   1.000000  0.282678  0.662573
    City_N                    0.039672       0.004462  -0.011100      0.021106        -0.103554  0.305708    -0.116663  0.274515  0.282426    0.316091   0.037109        -0.247981         0.002546               0.003467 -0.103607      0.008023               -0.010280  -0.157804  0.185122        -0.171950        0.122258          -0.150678   0.282678  1.000000  0.341550
    Region_N                  0.037371      -0.001537  -0.063825     -0.007473         0.051434  0.525084    -0.062811  0.443472  0.460081    0.612007   0.048584        -0.216974        -0.001458               0.015772 -0.093961      0.020161               -0.004928  -0.088392  0.059788        -0.409303        0.273226          -0.193721   0.662573  0.341550  1.000000
    
    
    
    RMSE values for each variable:  {'Playtime': 1973.27834771793, 'Interruptions': 1.845749616324548, 'Join Time': 2.29533590634888, 'Buffer Ratio': 3.752383725453727, 'Connection Type': 4.656891397425496, 'Device': 3.7234283103860433, 'Device Type': 2.3125018585946937, 'Browser': 2.545649862818629, 'OS': 1.9460076443869307, 'OS Version': 16.011175093652824, 'Device ID': 511.74239167918165, 'Happiness Score': 4.356839744546628, 'Playback Stalls': 0.07319335699608348, 'Startup Error (Count)': 0.12121741568563121, 'Latency': 19827.84413455735, 'Crash Status': 0.12988804884375216, 'End of Playback Status': 0.29318975892334087, 'User_ID_N': 168.4403728796145, 'Title_N': 497.7516045157842, 'Device_Vendor_N': 4.859356290457587, 'Device_Model_N': 31.487041492299486, 'Content_TV_Show_N': 649.6824248264899, 'Country_N': 2.6484208895447368, 'City_N': 110.74372276194134, 'Region_N': 0.2041694762577156}
    

From the RMSE result it can be seen variable with the best RMSE is Playback Stalls with RSME 0.07 and the worst is Latency with RMSE 19827.84. 

This differences happend because residual data in each features is also different. From the result we can say that Latency have more residual than Playback Stall

# Plot actual and forecast data


```python
# Plot actual and forecast data
n_variables = len(test_data.columns)
n_cols = 2
n_rows = (n_variables + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 6 * n_rows))

for ax, var in zip(axes.flatten(), test_data.columns):
    ax.plot(train_data.index, train_data[var], label='Train')
    ax.plot(test_data.index, test_data[var], label='Actual')
    ax.plot(forecast_df.index, forecast_df[var], label='Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel(var)
    ax.set_title(f'{var} - Actual vs Forecast')
    ax.legend()

# Remove any unused subplots
for ax in axes.flatten()[n_variables:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()

```


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_91_0.png)
    


from the graph, it can be seen the forecast have strong differences from the actual value. It is happend because the dataset is residual and have many anomaly or irregural fluctuation over the time.



**5.1.1 Change Detection Using CUSUM Detector Method**

CUSUM (Cumulative Sum) detector is a statistical algorithm used for detecting changes or shifts in the mean or variance of a time series data.

The basic idea behind the CUSUM algorithm is to calculate the cumulative sum of the deviations of the data points from the mean or target value. If the sum exceeds a certain threshold, it indicates a change point.

We perform change detection between actual and forecast data using CUSUM method


```python
from sklearn.metrics import mean_absolute_error

# Align the indexes of test_data and forecast_df
aligned_test_data, aligned_forecast_df = test_data.align(forecast_df, axis=0)
```


```python
import numpy as np

def cusum(series, threshold):
    mean = series.mean()
    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))
    
    for t in range(1, len(series)):
        cusum_pos[t] = max(0, cusum_pos[t-1] + series[t] - mean - threshold)
        cusum_neg[t] = max(0, cusum_neg[t-1] - series[t] + mean - threshold)
    
    return cusum_pos, cusum_neg

# Calculate the difference between actual and forecast data
difference = aligned_test_data - aligned_forecast_df

# Set the threshold as the Mean Absolute Error (MAE)
threshold = {var: mean_absolute_error(aligned_test_data[var], aligned_forecast_df[var]) for var in test_data.columns}

# Apply CUSUM method and find significant changes
change_detection = {}

for var in test_data.columns:
    cusum_pos, cusum_neg = cusum(difference[var], threshold[var])
    change_detection[var] = (cusum_pos > 0) | (cusum_neg > 0)

# Print the significant changes
print("Significant changes:")
for var in test_data.columns:
    changes = change_detection[var][change_detection[var]]
    print(f"{var} ({len(changes)} changes):")
    print(changes)

```

    Significant changes:
    Playtime (6358 changes):
    [ True  True  True ...  True  True  True]
    Interruptions (5629 changes):
    [ True  True  True ...  True  True  True]
    Join Time (8824 changes):
    [ True  True  True ...  True  True  True]
    Buffer Ratio (7218 changes):
    [ True  True  True ...  True  True  True]
    Connection Type (8598 changes):
    [ True  True  True ...  True  True  True]
    Device (6949 changes):
    [ True  True  True ...  True  True  True]
    Device Type (7397 changes):
    [ True  True  True ...  True  True  True]
    Browser (7278 changes):
    [ True  True  True ...  True  True  True]
    OS (7211 changes):
    [ True  True  True ...  True  True  True]
    OS Version (6959 changes):
    [ True  True  True ...  True  True  True]
    Device ID (14676 changes):
    [ True  True  True ...  True  True  True]
    Happiness Score (16337 changes):
    [ True  True  True ...  True  True  True]
    Playback Stalls (6526 changes):
    [ True  True  True ...  True  True  True]
    Startup Error (Count) (7082 changes):
    [ True  True  True ...  True  True  True]
    Latency (12026 changes):
    [ True  True  True ...  True  True  True]
    Crash Status (9317 changes):
    [ True  True  True ...  True  True  True]
    End of Playback Status (6740 changes):
    [ True  True  True ...  True  True  True]
    User_ID_N (12688 changes):
    [ True  True  True ...  True  True  True]
    Title_N (11441 changes):
    [ True  True  True ...  True  True  True]
    Device_Vendor_N (7524 changes):
    [ True  True  True ...  True  True  True]
    Device_Model_N (10606 changes):
    [ True  True  True ...  True  True  True]
    Content_TV_Show_N (7323 changes):
    [ True  True  True ...  True  True  True]
    Country_N (7120 changes):
    [ True  True  True ...  True  True  True]
    City_N (14356 changes):
    [ True  True  True ...  True  True  True]
    Region_N (8017 changes):
    [ True  True  True ...  True  True  True]
    


```python
# Plot the CUSUM change detection results
n_variables = len(test_data.columns)
n_cols = 2
n_rows = (n_variables + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 6 * n_rows))

for ax, var in zip(axes.flatten(), test_data.columns):
    ax.plot(aligned_test_data.index, aligned_test_data[var], label='Actual')
    ax.plot(aligned_forecast_df.index, aligned_forecast_df[var], label='Forecast')
    
    significant_changes = np.where(change_detection[var])[0]
    for change_idx in significant_changes:
        change_date = aligned_test_data.index[change_idx]
        ax.plot(change_date, aligned_test_data[var].iloc[change_idx], 'ro', label='Significant Change' if change_idx == significant_changes[0] else None)

    ax.set_xlabel('Date')
    ax.set_ylabel(var)
    ax.set_title(f'{var} - Actual vs Forecast with CUSUM Change Detection')
    ax.legend()

# Remove any unused subplots
for ax in axes.flatten()[n_variables:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()

```

    <ipython-input-39-c58af17bb7c7>:26: UserWarning:
    
    Creating legend with loc="best" can be slow with large amounts of data.
    
    /usr/local/lib/python3.9/dist-packages/IPython/core/pylabtools.py:151: UserWarning:
    
    Creating legend with loc="best" can be slow with large amounts of data.
    
    


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_98_1.png)
    


From the graph, it can be seen there are change-point detected in every features. Sudden change-point appear on almost every point on every features. We can assume this is happend because the dataset itself is residual data with many irregular data fluctuates over the time

**5.1.2 windown Based**

Window-based change point detection is used to perform fast signal segmentation, we use window-based algorithm implemented in ruptures library. The algorithm uses two windows which slide along the time series data. The statistical properties of the signals within each window are compared with a discrepancy measure. If the sliding windows both fall into a segment, their statistical properties are similar and the discrepancy between the first window and the second window is low. If the sliding windows fall into two dissimilar segments, the discrepancy is significantly higher, suggesting that the boundary between windows is a change point. A sequential peak search is performed on the discrepancy curve in order to detect change points.

The benefits of window-based segmentation includes low complexity, also it can extend any single change point detection method to detect multiple changes points and that it can work whether the number of regimes is known beforehand or not.



```python
!pip install ruptures
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting ruptures
      Downloading ruptures-1.1.7-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m19.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: scipy in /usr/local/lib/python3.9/dist-packages (from ruptures) (1.10.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.9/dist-packages (from ruptures) (1.24.2)
    Installing collected packages: ruptures
    Successfully installed ruptures-1.1.7
    


```python
import ruptures as rpt

# load the actual and forecast data
actual_data = test_data
forecast_data = forecast_df

# convert data to NumPy arrays if they are Pandas DataFrames
if isinstance(actual_data, pd.DataFrame):
    actual_data = actual_data.values
    variable_names = test_data.columns
if isinstance(forecast_data, pd.DataFrame):
    forecast_data = forecast_data.values

# apply the window-based search method to detect change points
model = "l2"
algo = rpt.Window(width=40, model=model).fit(forecast_data)
result = algo.predict(n_bkps=14)
```


```python
# plot the result for each variable
num_variables = actual_data.shape[1]

# determine the number of rows and columns for the grid
nrows, ncols = num_variables // 2 + num_variables % 2, 2

# set the figure size
fig, ax = plt.subplots(nrows, ncols, figsize=(12 * n_cols, 6 * n_rows), sharex=True)

for i in range(nrows):
    for j in range(ncols):
        idx = i * ncols + j

        if idx < num_variables:
            ax[i, j].plot(actual_data[:, idx], label='Actual Data', alpha=0.7)
            for cp in result:
                ax[i, j].axvline(cp, color='r', linestyle='--', linewidth=1, alpha=0.7)
            ax[i, j].legend(loc='upper left')
            ax[i, j].set_ylabel(variable_names[idx])

# set the x-axis label for the last row
for j in range(ncols):
    ax[-1, j].set_xlabel('Time')

# set the title
fig.suptitle('Change Point Detection between Actual and Forecast Data')

plt.show()

```


    
![png](Change_Detection_on_CDN_Dataset_files/Change_Detection_on_CDN_Dataset_104_0.png)
    


It can be seen window-based method not show the expected result. It can detect significant change in the given data but not able to detect every change point as it can see the result is same in every features. It caused window-base method not able to perform change detection in multivariate data. It useful for univariate time series data not multivariate time series data.

From all of the result, it shows bad latency causes users to stop playing the video. This event causes residual in the data and thus, spread residual data to every features. From change detection it also shows this residual appear almost every time which shows there is problem with latency in the network. 


```python

```
