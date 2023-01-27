import tensorflow as tf
import glob
from pathlib import Path
from preprocessing import preprocessing
from predict import predict
import pandas as pd
import wave
from pydub import AudioSegment
import os
import re
import pandas as pd
# read audio file
from pydub import AudioSegment
# print os.environ['PATH']

# input files
inputFolder = "input/2min/"
preprocessedFolder = inputFolder.replace('input','preprocessed')
outputFolder = inputFolder.replace('input','output')

