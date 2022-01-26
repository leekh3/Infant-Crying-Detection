# Infant Crying Detection
This repository contains code to run an infant crying detection model for continuous audio in real-world environments, as described in this [paper](https://arxiv.org/abs/2005.07036).

## Citation Information
X. Yao, M. Micheletti, M. Johnson, E. Thomaz, and K. de Barbaro, "Infant Crying Detection in Real-World Environments," in ICASSP 2022 (Accepted)


## Models and Main Package Versions
Trained deep spectrum model can be found at: https://utexas.box.com/s/64ecwy5wo0zzla4sax3j30dog0f4k8kv  
Trained SVM model is in this repository: svm.joblib  

### Versions
python3/3.6.3  
tensorflow-gpu==1.13.2  
scikit-learn==0.23.0   
pyAudioAnalysis==0.3.7  
librosa==0.8.1  


# Code
There are two scripts: *preprocessing.py* and *predict.py*.

*preprocessing.py* aims to get rid of the seconds that are definitely not infant crying using frequency information. It reads an audio file and outputs a csv file containing the start_time (seconds) and end_time (seconds) of audio where there is some energy for signals higher than 350Hz. Change input/output filennames here:
```
audio_filename = 'P34_2.wav'
output_file = "preprocessed.csv"
```


*predict.py* gives predictions of crying/not crying at every second. It reads an andio file and the preprocessed csv file and outputs a csv file containing the predictions at each second with its timestamp. Change input/output filenames here:

```
preprocessed_file = "preprocessed.csv"
audio_filename = "P34_2.wav"
output_file = "predictions.csv"
```


## Other resources
1. M. Micheletti, X. Yao, M. Johnson, and K. de Barbaro, "Validating a Model to Detect Infant Crying from Naturalistic Audio," (Under Review)

2. HomeBank English deBarbaro Cry Corpus (https://homebank.talkbank.org/access/Secure/deBarbaroCry-protect/deBarbaroCry.html)  
	It contains part of the RW-Filt dataset that was used to create the model as 2 out of 24 participants did not give us permission to share their data.  
	To protect the privacy of participants, all crying episodes were cut into five second segments (with four-second overlap between neighboring segments). An equal length and number of five second segments of non-cry data was randomly selected from the same recording. The complete dataset totals 61.3h of labelled data with over seven hours of unique annotated crying data. 


