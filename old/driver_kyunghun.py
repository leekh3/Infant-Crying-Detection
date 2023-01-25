import glob
import librosa.display
import numpy as np

inDataFolder = '/Users/jimmy/data/'
nSeg = 5
folderList = glob.glob(inDataFolder + '/*/',recursive = False)
folderList.sort()

x = []
y = []
for inFolder in folderList:
    inFiles = glob.glob(inFolder + '/*.wav')
    for i,inFile in enumerate(inFiles):
        xdata,_ = librosa.load(inFile)
        xLen = int(xdata.shape[0] / nSeg)
        for j in range(nSeg):
            x.append(xdata[j*xLen:(j+1)*xLen])
            if "notcry" in inFile:
                y.append(0)
            else:
                y.append(1)


a = np.array(x)
# inFile = inFiles[0]# y, _ = librosa.load(inFile)

