# HAAR Cascade Training Attempts

## Attempt #1 (training_01)
Parameters:
P: 1000
N: 1200
Final: P's combined with N'
samples.vec (24x24)
Size 24x24

---
PARAMETERS:
cascadeDirName: p/
vecFileName: positives.vec
bgFileName: negatives.txt
numPos: 1000
numNeg: 1200
numStages: 20
precalcValBufSize[Mb] : 1024
precalcIdxBufSize[Mb] : 1024
acceptanceRatioBreakValue : -1
stageType: BOOST
featureType: HAAR
sampleWidth: 24
sampleHeight: 24
boostType: DAB
minHitRate: 0.995
maxFalseAlarmRate: 0.5
weightTrimRate: 0.95
maxDepth: 1
maxWeakCount: 100
mode: ALL
Number of unique features given windowSize [24,24] : 261600
---
STAGES: 19
OVERALL DURATION: 0 days 3 hours 51 minutes 51 seconds
RESULTS: Unable to detect!


## Attempt #2 (training_02)
training_02:
Parameters:
P: 1000
N: 1200
Final: P's combined with N' + all P's
samples.vec (24x24)
Size 24x24

---
PARAMETERS:
cascadeDirName: p/
vecFileName: positives.vec
bgFileName: negatives.txt
numPos: 1000
numNeg: 1200
numStages: 20
precalcValBufSize[Mb] : 1024
precalcIdxBufSize[Mb] : 1024
acceptanceRatioBreakValue : -1
stageType: BOOST
featureType: HAAR
sampleWidth: 24
sampleHeight: 24
boostType: DAB
minHitRate: 0.995
maxFalseAlarmRate: 0.5
weightTrimRate: 0.95
maxDepth: 1
maxWeakCount: 100
mode: ALL
Number of unique features given windowSize [24,24] : 261600

---
STAGES: ?
OVERALL DURATION: ?
RESULTS: ?

## Attempt #3 (training_03)
training_03:
Parameters:
P: 3000
N: 1200
Final: P's combined with N' + all P's
samples.vec (50x50)
Size 50x50
