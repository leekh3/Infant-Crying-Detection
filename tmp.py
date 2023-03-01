import random
from pydub import AudioSegment

# Define constants
ONE_HOUR = 60 * 60 * 1000  # Length of 1 hour segment in milliseconds

# Load the WAV file
wav_file = AudioSegment.from_wav("/Users/leek13/data/LENA/1180/e20171121_094647_013506.wav")

# Calculate the maximum start position for the extracted segment
max_start_pos = len(wav_file) - ONE_HOUR

# Generate a random start position within this range
start_pos = random.randint(0, max_start_pos)

# Calculate the end position for the extracted segment
end_pos = start_pos + ONE_HOUR

# Extract the 1-hour segment from the WAV file
extracted_segment = wav_file[start_pos:end_pos]

# Save the extracted segment to a new WAV file
extracted_segment.export("test.wav", format="wav")