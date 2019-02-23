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
