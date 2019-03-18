from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import pandas as pd
import keras
from hyperas.distributions import uniform, choice
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
## For reproducibility
from numpy.random import seed
seed(10241996)

def data():
    train_values = pd.read_csv('train_values.csv', index_col='patient_id')
    train_labels = pd.read_csv('train_labels.csv', index_col='patient_id')
    #train_labels.heart_disease_present.value_counts().plot.bar(title='Number with Heart Disease')
    #selected_features = ['age', 
    #                     'sex', 
    #                     'max_heart_rate_achieved', 
    #                     'resting_blood_pressure']
    selected_features =['slope_of_peak_exercise_st_segment',
    'resting_blood_pressure',
    'chest_pain_type',
    'num_major_vessels',
    'fasting_blood_sugar_gt_120_mg_per_dl',
    'resting_ekg_results',
    'serum_cholesterol_mg_per_dl',
    'oldpeak_eq_st_depression',
    'sex',
    'age',
    'max_heart_rate_achieved',
    'exercise_induced_angina']
    train_values_subset = train_values[selected_features]
    predictors =train_values_subset
    target = train_labels.heart_disease_present
    x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.10,random_state=0)

    return x_train, y_train, x_test, y_test


def create_model(x_train,y_train,x_test,y_test):
	"""
	Keras model function.
	"""

	inshape = 12
	outshape = 1
	min_hlayers=3


	model = Sequential()
	for i in range(min_hlayers):
		if i==0:
			model.add(Dense({{ choice(range(50)) }},input_shape=(inshape,)))
			model.add(Activation({{ choice(['relu','sigmoid']) }})) ## Choose between relu or signmoid activation
			model.add(Dropout({{ uniform(0,1) }})) ## Choose dropout value using uniform distribution of values from 0 to 1
		else:
			model.add(Dense({{ choice(range(50)) }}))
			model.add(Activation({{ choice(['relu','sigmoid']) }}))
			model.add(Dropout({{ uniform(0,1) }}))

	model.add(Dense(outshape))
	model.add(Activation({{ choice(['relu','sigmoid']) }}))

	## Hyperparameterization of optimizers and learning rate
	_adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
	_rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
	_sgd = keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})

	opt_choiceval = {{ choice( ['_adam', '_rmsprop', '_sgd'] ) }}

	if opt_choiceval == '_adam':
		optim = _adam
	elif opt_choiceval == '_rmsprop':
		optim = _rmsprop
	else:
		optim = _sgd
	
	model.compile(loss='mean_absolute_error', metrics=['mse'],optimizer=optim)
	model.fit(x_train, y_train,
		batch_size=100,
		epochs=5,
		verbose=2,
		validation_data=(x_test, y_test))

	score, acc = model.evaluate(x_test, y_test)
	predicted = model.predict(x_test)

	## Print validation set
	# for i in range(5):
	# 	print("Pred: ",predicted[i], " Test: ",y_test[i])
	print('Test accuracy:', acc)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def main():
		
	trials =Trials()
	best_run, best_model = optim.minimize(model=create_model,
	                                      data=data,
	                                      algo=tpe.suggest,
	                                      max_evals=3,
	                                      trials=trials)
	x_train, y_train, x_test, y_test = data()
	print("\n >> Hyperparameters  ")
	for t in best_run.items():
		print("[**] ",t[0],": ", t[1])

	print("\nSaving model...")
	model_json = best_model.to_json()
	with open("model_num.json","w") as json_file:
		json_file.write(model_json)
	best_model.save_weights("model_num.h5")

	



if __name__ == "__main__":
	main()
	print("done..")
