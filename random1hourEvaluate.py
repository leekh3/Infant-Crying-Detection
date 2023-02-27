import pandas as pd
import random

df = pd.read_csv('/Users/jimmy/data/LENA/ListofLENAParticipants.csv',header=None)
idxList = df[0].tolist()
idxList.sort()
nums = [i for i in range(302)]
random.shuffle(nums)
nums = nums[:100]

randomList = []
for i in range(len(nums)):
    randomList.append(idxList[nums[i]])

dfOut = pd.DataFrame()
dfOut['subject'] = randomList
dfOut.to_csv('output_detection_for_elan/random100.csv', sep=',')

