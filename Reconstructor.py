import BVH
import struct
import copy
import sys
import os
import math
import numpy as np
from nn import NeuralNetwork

def getLength(a, b):
    return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]))

def preprocessModel(posBatch, isRefine=True, isNormalize=True):
    ###preprocess JointPosition into 15*3 form & Normalize
    ###Input : jointPos (19 * 3)
    ###jointIndex should not point index number higher than itself
    ###Result :

    # 1. Refine Data into "jointDB" form
    jointIndex = [-1, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 0, 12, 13]

    stanfordIndex= [8, 1, 5, 6, 7, 2, 3, 4, -1, 0, -1, 13, 14, -1, 10, 11, 9, 12, -1]


    if isRefine:
        refinedPosBatch = []
        for i in range(0, len(posBatch)):
            currPos = np.array((15, 3))
            for j in range(0, len(stanfordIndex)):
                if stanfordIndex[j] == -1:
                    continue

                currPos[stanfordIndex[j]][0] = float(posBatch[i][j][0])
                currPos[stanfordIndex[j]][1] = float(jointPos[i][j][1])
                currPos[stanfordIndex[j]][2] = float(jointPos[i][j][2])
            refinedPosBatch.append(currPos)
    else:
        refinedPosBatch = []

    # 2. Normailze (Transition & Scaling)
    if isNormalize:
        mean_list = np.mean(posBatch, axis=0)
        std_list = np.std(posBatch, axis=0)

        for i in range(len(posBatch)):
            currPos = np.zeros((15, 3))
            for j in range(1, len(jointIndex)):
                currPos[j][0] = (posBatch[i][j][0] - mean_list[j][0]) / std_list[j][0]
                currPos[j][1] = (posBatch[i][j][1] - mean_list[j][1]) / std_list[j][1]
                currPos[j][2] = (posBatch[i][j][2] - mean_list[j][2]) / std_list[j][2]

            refinedPosBatch.append(currPos)
    else:
        mean_list = np.zeros((15, 3))
        std_list = np.zeros((15, 3))

    #print "inside preprocess", refinedPos

    return np.array(refinedPosBatch), mean_list, std_list

def reconstructModel(network, jointPos, missingMarker=None):
    resultPos = network.reconstruct(jointPos, 1/15.0)
    
    '''if missingMarker is not None:
        for i in range(0, len(resultPos)):
            if missingMarker[i] == True:
                resultPos[i] = resultPos[i]
            else:
                resultPos[i] = jointPos[i]  
    '''

    return resultPos

def postprocessModel(originalPos, jointPos, trans, scale):
    for jointCoor in jointPos:
        jointCoor[0] = jointCoor[0]*scale + trans[0]
        jointCoor[1] = jointCoor[1]*scale + trans[1]
        jointCoor[2] = jointCoor[2]*scale + trans[2]

    #print "Scale : ", scale
    #print "Trans : ", trans
    #print jointPos

    resultPos = [[0,0,0]]*19

    revIndices=[9, 1, 5, 6, 7, 2, 3, 4, 0, 16, 14, 15, 17, 11, 12]

    for i in range(0, len(revIndices)):
        resultPos[revIndices[i]] = jointPos[i]

    abbvPos=[8, 10, 13, 18]

    for eachPos in abbvPos:
        resultPos[eachPos] = originalPos[eachPos]

    return resultPos


def main(ckptname, test_dbname, result_dbname, recon):


    jointCount, fileNameList, jointPosBatch = BVH.load_stanford(test_dbname)

    usableFileList = []
    resultPosBatch = []
    for jointPos, fileName in zip(jointPosBatch, fileNameList):

        #print "LoadBVH"
        #for pos in jointPos:
        #    print pos
        isUsable, refinedPos, missingMarker, trans, scale = recon.preprocessModel(jointPos)
        if not isUsable:
            continue

        #print "PreProcess"
        #for pos in refinedPos:
        #    print pos

        resultPos = recon.reconstructModel(refinedPos, missingMarker)

        #print "Reconstruct"
        #for pos in resultPos:
        #    print pos

        resultPos = recon.postprocessModel(jointPos, resultPos, trans, scale)

        #print "PostProcess"
        #for pos in resultPos:
        #    print pos

        resultPosBatch.append(resultPos)
        usableFileList.append(fileName)

    BVH.save_stanford(result_dbname, usableFileList, resultPosBatch)

if __name__=="__main__":
    argv = sys.argv
    if not os.path.isfile(argv[1]):
        print "No Network Found!"
    elif not os.path.isdir(argv[2]):
        print "Input Should be a Directory!"
    #elif os.path.isfile(sys.argv[3]):
    #    print "Output Already Exists!"
    else:
        recon = Reconstructor(argv[1])
        if not os.path.isdir(argv[3]):
            os.makedirs(argv[3])
        #if argv[4] Exists:
        #    isOnlyMissingMarker = argv[4]
        for file in os.listdir(argv[2]):
            inputName = argv[2]+"/"+file
            networkName = argv[1][:argv[1].find(".")]
            outputName = argv[3]+"/"+networkName+"_"+file
            print "Input : ", inputName
            main(ckptname=argv[1], test_dbname=inputName, result_dbname=outputName, recon=recon)
            print "Output : ", outputName , " Complete!..."
