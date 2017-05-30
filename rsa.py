'''
Randomized Search Algorithm: Optimizing with Randomization.

Description:
	Randomly samples across the input space, and then gradually narrows in
	on the regions that contain optimal values by means of a shrinking
	Gaussian distribution.

Function to be maximized (The inverse, in the case of a loss function):
function = lambda input: input[0]*input[1]

Usage: rsa.run(dimension, extent, function, epochs, nb_optimal_values)

'''


import numpy as np

def sample(size, dimension, extent):
	the_sample = extent* np.random.rand(size, dimension)
	return the_sample

def evaluate(sample, function):
	values = []
	for item in sample:
		values.append(function(item))

	return values

def iterate(sample, values, progression_window):
	dimension = sample.shape[1]
	procreation_parameter = 5

	sort_index = np.argsort(values).tolist()
	sort_index.reverse() #Optimization
	sorted_sample = sample[sort_index]
	pruned_sample = sorted_sample[:progression_window]

	new_sample = []
	deviation = np.std(pruned_sample)/30 #Breeding restriction becomes more stringent with time.
	# print 'Breeding Laxity:', deviation

	for item in pruned_sample:
		new_subsample = np.random.normal(item, deviation, [procreation_parameter, dimension])
		for progeny in new_subsample:
			new_sample.append(progeny)
		new_sample.append(item)
	return np.array(new_sample)

def run(dimension, extent, function, epochs, nb_optimal_values):
	size = 100
	sample_instance = sample(size, dimension, extent)

	for epoch in range(epochs):
		values = evaluate(sample_instance, function)
		sample_instance = iterate(sample_instance, values, progression_window= 5)
	return sample_instance[:nb_optimal_values]
