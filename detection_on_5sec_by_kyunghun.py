def detection_on_5sec_by_kyunghun(inFolder):
    import librosa
    import numpy as np
    import glob
    from os.path import expanduser
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    from scipy import signal
    import numpy as np
    import librosa
    import tensorflow as tf

    # find all folders
    # home = expanduser("~")
    # inFolders = glob.glob(home + "/data/LENA/*/AN1/segmented_2min/")
    # inFolder = inFolders[0]

    # find the list of wav files.
    inFiles = glob.glob(inFolder + '/*.wav')
    inFiles.sort()

    # Load audio file
    detected_count = 0
    for audio_file in inFiles:
        import numpy as np
        import scipy.signal as sig
        from scipy.io import wavfile

        # Define the bandpass filter to extract frequency components in the range of infant crying
        low_freq = 250  # Hz
        high_freq = 3000  # Hz
        nyquist_freq = 22050 // 2  # Hz


        # Define a function for detecting infant crying
        def detect_infant_crying(wav_file_path):
            # Load the audio file
            fs, data = wavfile.read(wav_file_path)
            if data.ndim > 1:  # if the audio file is stereo, convert to mono
                data = np.mean(data, axis=1)
            data = data.astype(np.float32)

            # Apply the bandpass filter to extract frequency components in the range of infant crying
            b, a = sig.butter(4, [low_freq / nyquist_freq, high_freq / nyquist_freq], btype='band')
            filtered_data = sig.filtfilt(b, a, data)

            # Compute the energy of the filtered signal using a sliding window
            window_size = int(fs * 0.1)  # 100 ms window
            step_size = int(fs * 0.05)  # 50 ms step
            energy = np.array([np.sum(filtered_data[i:i + window_size] ** 2) for i in
                               range(0, len(filtered_data) - window_size, step_size)])

            # Compute the short-term average and long-term average of the energy
            short_term_avg_energy = np.mean(energy)
            long_term_avg_energy = np.mean(energy[-10:])

            # Compute the ratio of short-term average energy to long-term average energy
            energy_ratio = short_term_avg_energy / long_term_avg_energy

            return energy_ratio > 1.5


        # Example usage
        file_path = audio_file
        is_crying = detect_infant_crying(file_path)
        if is_crying:
            print('The audio file contains infant crying.')
            detected_count += 1
        else:
            print('The audio file does not contain infant crying.')
            continue
    print(detected_count)
    return detected_count