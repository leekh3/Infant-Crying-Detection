def segment_into_5sec(input_wav_path):
    import glob
    from pydub import AudioSegment
    import os

    # specify the directory containing the WAV files
    directory = "tmp"

    # use glob to find all WAV files in the directory
    wav_files = glob.glob(os.path.join(directory, "*.wav"))

    # loop over each WAV file and delete it
    for wav_file in wav_files:
        os.remove(wav_file)

    # set the duration of each segment in milliseconds (5 seconds = 5000 ms)
    segment_duration = 5000

    # specify the input WAV file path
    # input_wav_path = "path/to/input/file.wav"
    # input_wav_path = '/Users/leek13/data/LENA_random_1hour/1180_LENA/AN1/e20171121_094647_013506.wav'

    # load the input WAV file as an AudioSegment object
    audio = AudioSegment.from_wav(input_wav_path)

    # calculate the number of segments in the input file
    num_segments = len(audio) // segment_duration

    # create a directory to store the output segments
    output_dir = "tmp"
    os.makedirs(output_dir, exist_ok=True)

    # export each segment as a separate WAV file
    for i in range(num_segments):
        # calculate the start and end time of the current segment
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration

        # extract the segment as a new AudioSegment object
        segment = audio[start_time:end_time]

        # export the segment as a WAV file with a unique filename
        output_filename = f"segment_{i+1}.wav"
        output_path = os.path.join(output_dir, output_filename)
        segment.export(output_path, format="wav")

