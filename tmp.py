from scipy.signal import savgol_filter
S_sum = [0,0,1,1,0,1,1,1,0,0,0,1,1,1,1,0,0,0,0,1,1]
filted = savgol_filter(S_sum, len(S_sum), 3)

