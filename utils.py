def scaler(min, max):
	def _scaler(x):
		return (x - min) / (max - min)
	return _scaler

def rescaler(min, max):
	def _rescaler(x):
		return x * (max - min) + min
	return _rescaler
