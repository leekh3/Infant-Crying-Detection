def preprocessing_kyunghun(audio_filename,output_file):
	audio_filename = '/Users/jimmy/data/LENA/1198_LENA/AN1/segmented_2min/0.wav'
	import librosa
	import numpy as np
	import librosa.display
	import csv
	from scipy.signal import savgol_filter
	from scipy import signal
	import matplotlib.pyplot as plt

	##Read audio file
	y, sr = librosa.load(audio_filename)
	duration = librosa.get_duration(y=y, sr=sr)

	sos = signal.butter(10, 600, 'hp', fs=sr, output='sos')
	y = signal.sosfilt(sos, y)
	# We'll need IPython.display's Audio widget

	x = y
	x[abs(x) < 0.1] = 0
	x[abs(x) > 0] = 1

	freq = int(len(y) / 120)
	# i = 1
	# firstIdx = i*freq
	# lastIdx = (i+1)*freq
	# winSum = sum(x[firstIdx:lastIdx])
	# print(winSum)
	times = []
	for i in range(120):
		# i = 0
		firstIdx = i * freq
		lastIdx = (i + 1) * freq
		winSum = -sum(x[firstIdx:lastIdx])
		times.append(int(winSum))
	# print(winSum)

	filted = savgol_filter(times, 3, 1)
	filted[filted < 0] = -1
	filted[filted > 0] = -1
	filted[filted == 0] = 0
	times = filted
	label = 1
	lastIdx = 0
	rs = []

	for i in range(len(times)):
		if times[i] >= 0 or i < lastIdx:
			continue

		r = [i, i]
		lastIdx = i
		for j in range(i, len(times)):
			if times[j] < 0:
				times[j] = label
				r[-1] = j
				lastIdx = j
				if j == len(times) - 1:
					rs.append(r)
			else:
				label += 1
				rs.append(r)
				break

	output_file = 'test.csv'
	##write output into a file
	with open(output_file, 'w', newline = '') as f:
		writer = csv.writer(f)
		writer.writerows(rs)

	return rs

