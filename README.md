# **Datascience cheat sheet IALAB GARAGEISEP**


![alt text](https://www.isep.fr/wp-content/uploads/2017/12/garage-isep-e1513856613884.jpg "Logo Title Text 1")


## Data preprocessing
##### Normalize dataset

```python
df[columns_to_norm] = df[columns_to_norm].apply(lambda x: (x - min(x)) / (x.max()-x.min()))
```
##### Separate train/test set
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
```
##### Confusion Matrix
```Python
from sklearn import metrics 
 
expected = df['Y']
predicted = gnbModel.predict(df[['Col1','Col2']]) 
print(metrics.classification_report(expected, predicted)) 
```
##### Hide warnings
```Python
import warnings
warnings.filterwarnings('ignore')
```
##### Convert object column to numeric and replace NaN
```Python
df['Col'] = pd.to_numeric(df['Col'].replace(np.NaN,'0'))
```
##### Loop into two lists with enumerate
```Python
for i,(item1,item2) in enumerate(zip(df['Col1'],df['Col2'])):
    ...
```
##### Select some rows in a dataset depending on a value
```Python
df_is_win = df.loc[df['Col'] == val]
```

##### Read csv
```Python
pd.read_csv(“csv_file”)
```
##### Export dataset to csv
```Python
df.to_csv("data.csv", sep=",", index=False)
```
##### Basic dataset feature info
```Python
df.info()
```
##### Basic dataset insights
```Python
print(df.describe())
```
##### See dataframe columns
```Python
df.columns
```
##### Drop missing value
```Python
df.dropna(axis=0, how='any')
```
##### Replace value
```Python
df.replace(to_replace=None, value=None)
```
##### Check for NaN
```Python
pd.isnull(object)
```
##### Drop a column
```Python
df.drop('column', axis=1)
```
##### Rename column
```Python
df.rename(columns = {df.columns[2]:'size'}, inplace=True)
```
##### Get the unique entries of a column
```Python
df["name"].unique()
```
##### Summary information about your data
```Python
# Sum of values in a data frame
df.sum()

# Lowest value of a data frame
df.min()

# Highest value
df.max()

# Index of the lowest value
df.idxmin()

# Index of the highest value
df.idxmax()

# Statistical summary of the data frame, with quartiles, median, etc.
df.describe()

# Average values
df.mean()

# Median values
df.median()

# Correlation between columns
df.corr()

# To get these values for only one column, just select it like this#
df["size"].median()
```
##### Sorting your data
```Python
df.sort_values(ascending = False)
```
##### One hot encoder with pandas
```Python
col_encoded = pd.get_dummies(df['Col'])
```

***



## Tensorflow

##### Addition
```python
a = tf.constant(10)
b = tf.constant(15)

with tf.Session() as sess:
    result = sess.run(a + b)
print(result)
```

##### Generate tensor
```python
fill_mat = tf.fill((4,4),10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
my_randn = tf.random_normal((4,4), mean=0, stddev=1.0)
myrandu = tf.random_uniform((4,4), minval=0, maxval=1)
```

##### Interactive Session
```python
my_ops = [a, fill_mat, myzeros, my_randn]
sess = tf.InteractiveSession()
for op in my_ops:
    print(sess.run(op), '\n')
```
##### Placeholders/Variables
```python
sess = tf.InteractiveSession()

my_tensor = tf.random_uniform((4,4),0,1)
my_var = tf.Variable(initial_value=my_tensor)
my_ph = tf.placeholder(tf.float32)

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(my_var)
```

##### Linear Regression
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------- Generate data -------------------

x_data = np.linspace(0.0,10,1000000)
noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(x_data, columns=['X Data'])
y_df = pd.DataFrame(y_true, columns=['Y'])

my_data = pd.concat([x_df, y_df], axis=1)

# ---------------------- Variables ----------------------

batch_size = 16

m = tf.Variable(0.44)
b = tf.Variable(0.87)

X_placeholder = tf.placeholder(tf.float32, [batch_size])
Y_placeholder = tf.placeholder(tf.float32, [batch_size])

y_model = m * X_placeholder + b

init = tf.global_variables_initializer()

# -------------------- Loss Function ---------------------

error = tf.reduce_sum(tf.square(Y_placeholder-y_model))

# --------------------- Optimizer -------------------------
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# ---------------------- Train ----------------------------

with tf.Session() as sess:
    sess.run(init)
    batches = 10000
    for i in range (batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        feed = {X_placeholder:x_data[rand_ind],Y_placeholder:y_true[rand_ind]}
        sess.run(train, feed_dict=feed)
    final_slope, final_intercept = sess.run([m,b])

print("final_slope : ", final_slope)
print("final_intercept : ",final_intercept )

```
##### Linear Regression using API
```python
import tensorflow as tf
import pandas as pd
import numpy as np

x_data = np.linspace(0.0,10,1000000)
noise = np.random.randn(len(x_data))

y_data = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(x_data, columns=['X Data'])
y_df = pd.DataFrame(y_data, columns=['Y'])

my_data = pd.concat([x_df, y_df], axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
feats_col =[tf.feature_column.numeric_column('x', shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=feats_col)

input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train,batch_size=8, num_epochs=None,shuffle=True)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train,
                                                batch_size=8, num_epochs=1000,
                                                shuffle=False)

test_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_test}, y_test,
                                                batch_size=8, num_epochs=1000,
                                                shuffle=False)

estimator.train(input_fn = input_func, steps=1000)

train_metrics = estimator.evaluate(input_fn = train_input_func, steps=1000)

eval_metrics = estimator.evaluate(input_fn=test_input_func, steps=1000)

print('TRAINING DATA METRIC')
print(train_metrics)

print('TEST DATA METRIC')
print(eval_metrics)

brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)

list(estimator.predict(input_fn = input_fn_predict))
```

##### Bucket continuous values to categories
```python
age = tf.feature_column.numeric_column('Age')
bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])
```

##### Classifier API
```python
feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,age_bucket]
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=input_func, steps=1000)

#Test
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=1000,num_epochs=1,shuffle=False)
result = model.evaluate(eval_input_func)
#Predict
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=1000,num_epochs=1,shuffle=False)
list(model.predict(eval_input_func))
```

##### Multi Layer Perceptron API
```python
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

MLP_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10], feature_columns=feat_cols, n_classes=2)

MLP_model.train(input_fn=input_func, steps=1000)
```
##### Save Model
```python
saver = tf.train.Saver()
with tf.Session() as sess:
    ...
    saver.save(sess,'model/mymodel.ckpt')
```
##### Load Model
```python
with tf.Session() as sess:
    ...
    saver.restore(sess,'model/mymodel.ckpt')
```
##### Loss function with cross entropy
```python
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
```
##### MNIST with MLP
```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Placeholders
x = tf.placeholder(tf.float32, shape=[None,784])

#Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#GRAPH OPERATIONS
y = tf.matmul(x,W) + b

#LOSS FUNCTION
y_true = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

#OPTIMIZER
opt = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = opt.minimize(cross_entropy)

#SESSION
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y})
                                   
#EVALUATE THE MODEL
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(acc,feed_dict={x:mnist.test.images, y_true:mnist.test.labels}))
                                   
```
##### MNIST With CNN
```Python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# HELPER

# INIT WEIGHT

def init_weight(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

# INIT BIAS

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

# CONV2D

def conv2d(x, W):
    # x --> [batch, H, W, Channels]
    # W --> [feature_height, feature_width, feature IN, feature OUT]

    return tf.nn.conv2d(x, W,strides=[1,1,1,1], padding='SAME')

# POOLING

def max_pool_2by2(x):
    # x --> [batch, H, W, Channels]
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#CONVOLUTIONAL LAYER

def convolutional_layer(input_x, shape):
    W = init_weight(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)

def fully_connected_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weight([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer,W) + b

#Build the CNN

#Placeholder

x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

#Layers

x_image = tf.reshape(x,[-1,28,28,1])

convo_1 = convolutional_layer(x_image, shape=[5,5,1,32])
convo_1_pool = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pool, shape=[5,5,32,64])
convo_2_pool = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pool, [-1,7*7*64])
full_layer_one = tf.nn.relu(fully_connected_layer(convo_2_flat, 1024))

#Dropout

hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = fully_connected_layer(full_one_dropout, 10)

#LOSS Function

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits=y_pred))

#Optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

steps = 500
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):

        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y, hold_prob:0.5})

        if i%100==0:
            print("ON STEP : {}".format(i))
            print("ACCURACY : ")

            matches = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            print(sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0}),'\n')

```
##### One hot encoder 
``` Python
def one_hot_encode(vec, vals):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out
```
##### Recurrent neural network
``` Python
# Just one feature, the time series
num_inputs = 1
# 100 neuron layer, play with this
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1
# learning rate, 0.0001 default, but you can play with this
learning_rate = 0.0001
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 2000
# Size of the batch of data
batch_size = 1

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    for iteration in range(num_train_iteration):
        
        X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})
        
        if iteration % 100 == 0:
            
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    

```
##### Deep Nets with TF Abstractions
```Python
from sklearn.datasets import load_wine
wine_data = load_wine()
feat_data = wine_data['data']
labels = wine_data['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feat_data,
                                                    labels,
                                                    test_size=0.3,
                                                   random_state=101)
                                                   
from sklearn.preprocessing import MinMaxScaler
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow import estimator 

feat_cols = [tf.feature_column.numeric_column("x", shape=[13])]

deep_model = estimator.DNNClassifier(hidden_units=[13,13,13],
                            feature_columns=feat_cols,
                            n_classes=3,
                            optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01) )
                            
input_fn = estimator.inputs.numpy_input_fn(x={'x':scaled_x_train},y=y_train,shuffle=True,batch_size=10,num_epochs=5)
deep_model.train(input_fn=input_fn,steps=500)
input_fn_eval = estimator.inputs.numpy_input_fn(x={'x':scaled_x_test},shuffle=False)

preds = list(deep_model.predict(input_fn=input_fn_eval))

predictions = [p['class_ids'][0] for p in preds]

from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,predictions))
```
##### Keras Deep Neural Network with KFlod cross validation
```Python
# To get the dataset
# https://github.com/IALABGARAGE/transfermarkt-crawler-using-scrapy 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data():

    print("-------------- Chargement des données--------------")

    df_clean = pd.read_csv('dataset_football_cleaned.csv')
    df_clean['end_contract'] = df_clean['end_contract'].fillna(1)
    df_clean = df_clean.dropna()

    le_nation = OneHotEncoder(handle_unknown='ignore')
    le_ligue = OneHotEncoder(handle_unknown='ignore')
    le_equipe = OneHotEncoder(handle_unknown='ignore')
    le_poste = OneHotEncoder(handle_unknown='ignore')

    le_nation.fit(np.array((df_clean['nation'])).reshape(-1,1))
    le_ligue.fit(np.array((df_clean['league'])).reshape(-1,1))
    le_equipe.fit(np.array((df_clean['team'])).reshape(-1,1))
    le_poste.fit(np.array((df_clean['position'])).reshape(-1,1))


    df_clean['nation'] = le_nation.transform(np.array(df_clean['nation']).reshape(-1,1)).toarray()
    df_clean['league'] = le_ligue.transform(np.array(df_clean['league']).reshape(-1,1)).toarray()
    df_clean['team'] = le_equipe.transform(np.array(df_clean['team']).reshape(-1,1)).toarray()
    df_clean['position'] = le_poste.transform(np.array(df_clean['position']).reshape(-1,1)).toarray()


    X = df_clean.drop(['price'],axis=1)
    y = df_clean['price']

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print("-------------- Données chargées --------------")


    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, random_state=42)

    return X,y

def build_model(learn_rate=0.01, lbd=10e-10):

    # Initialising the ANN : création des différentes couches

    model = Sequential()

    model.add(Dense(units=256, activation='relu', input_dim=X.shape[1]))
    model.add(Dropout(0.2))

    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))


    opt = optimizers.RMSprop(lr=learn_rate, decay=lbd)

    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    # callbacks = [EarlyStopping(monitor='loss', patience=30)]

    return model

def plot_training(history):

    print(history.history.keys())

    # summarize history for loss
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])

    plt.title('model mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return

def predict():
    prediction = (pd.concat([pd.DataFrame(model.predict(X_test)),y_test.reset_index(drop=True)], axis=-1))
    return prediction


X,Y = load_data()

model = KerasClassifier(build_fn=build_model)

# define the grid search parameters
batch_size = [1,5]
epochs = [10]
learn_rate  = [0.005, 0.0005]
param_grid = dict(learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)
# summarize results


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




```
***

## Visualisation
##### Histogram with sns
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig,ax=plt.subplots(figsize=(18,8))
sns.distplot(df_train['X'].dropna(),color='darkred',bins=30,ax=ax)
```
##### Plot a curve
```python
plt.plot(ts_data.x_data, ts_data.y_true, label='Sin(t)')
plt.legend()
```
##### See histogram for each feature
```python
import pandas as pd
df.hist(figsize=(20,10),bins=200)
```
##### Pie chart
```python
df_train['feature'].value_counts().plot(kind='pie',subplots=True, figsize=(8, 8))
```
##### Correlation matrix
```python
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df.corr() 
fig,ax=plt.subplots(figsize=(18,8)) 
sns.heatmap(corr_matrix,annot=True)
```
***
### Anaconda
##### Create environnement
```python
conda create --name py3_tf_gpu python=3 anaconda pandas numpy scipy jupyter 
```
##### Activate environnement
```python
activate py3_tf_gpu 
```

##### Desactivate environnement
```python
deactivate
```
##### Remove environnement
```python
conda remove --name myenv --all
```
##### Install package
```python
pip install [package_name]
```

##### Upgrade package
```python
pip install [package_name] --upgrade
```
***

Sources:

- Matthieu Rousseau *Président IALAB at GarageISEP*  https://www.linkedin.com/in/matthieu-rousseau/
