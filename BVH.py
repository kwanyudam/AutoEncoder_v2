import numpy as np
import struct
import math
import copy
import re
import struct
import random

def getLength(a, b):
    return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]))

def standardization(jointData):

    for joint in jointData:
        #for jointCoor in joint:
        #    jointCoor[2] = jointCoor[2] * -10

        centerX = joint[0][0]
        centerY = joint[0][1]
        centerZ = joint[0][2]

        #scale = getLength(joint[0], joint[9]) / 0.2
        scale = getLength(joint[0], joint[1])

        for jointCoor in joint:
            jointCoor[0] = (jointCoor[0] - centerX) / scale
            jointCoor[1] = (jointCoor[1] - centerY) / scale
            jointCoor[2] = (jointCoor[2] - centerZ) / scale


def mirrorBVH(originalMesh):
    mesh = copy.copy(originalMesh)

    mesh[2] = originalMesh[5]
    mesh[5] = originalMesh[2]
    mesh[3] = originalMesh[6]
    mesh[6] = originalMesh[3]
    mesh[4] = originalMesh[7]
    mesh[7] = originalMesh[4]

    mesh[12] = originalMesh[9]
    mesh[9] = originalMesh[12]
    mesh[13] = originalMesh[10]
    mesh[10] = originalMesh[13]
    mesh[14] = originalMesh[11]
    mesh[11] = originalMesh[14]

    for jointCoor in mesh:
        jointCoor[0] = jointCoor[0] * -1  


    return mesh


#Input : FilePath
#Output : Training Set, Veryfing Set, Joint Names, Joint Parend Indices
def load_verify(filepath):
    print "Start Loading BVH saparating ... (default load)"
    f = open(filepath, 'r')

    trainingJointData = []
    verifyingJointData = []

    jointCount = f.readline()
    jointCount = int(jointCount)
    jointNames = f.readline()
    lines = f.readlines()

    trainingJointCount = 0
    verifyingJointCount = 0

    for line in lines:
        trim = line.find('>')
        line = line[trim+1:-2]
        eachjoint = line.split(' ')

        tpArray = []
        for val in eachjoint:
            tpval = float(val)
            tpArray.append(tpval)

        tpArray = np.reshape(tpArray, (jointCount, 3))

        mirrorArray = mirrorBVH(tpArray)

        if random.randrange(0,2) == 0:
            trainingJointData.append(tpArray)
            trainingJointData.append(mirrorArray)
            trainingJointCount+=1
        else:
            verifyingJointData.append(tpArray)
            verifyingJointData.append(mirrorArray)
            verifyingJointCount+=1

    print "BVH seperate Loading complete!"
    #joint Data in (N, 3) shape

    trainingJointData = np.array(trainingJointData)
    verifyingJointData = np.array(verifyingJointData)
    return trainingJointData, verifyingJointData, jointNames, [-1, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 0, 12, 13]


def load(filepath):
    print "Start Loading BVH ... (default load)"
    f = open(filepath, 'r')

    jointData = []

    jointCount = f.readline()
    jointCount = int(jointCount)
    jointNames = f.readline()
    lines = f.readlines()

    for line in lines:
        trim = line.find('>')
        line = line[trim+1:-2]
        eachjoint = line.split(' ')

        tpArray = []
        for val in eachjoint:
            tpval = float(val)
            tpArray.append(tpval)

        tpArray = np.reshape(tpArray, (jointCount, 3))

        mirrorArray = mirrorBVH(tpArray)
        

        #Mirror - change Joint Index between left & right

        jointData.append(tpArray)
        jointData.append(mirrorArray)

#    standardization(jointData)

    print "BVH Loading Complete!"
    return jointData, jointNames, [-1, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 0, 12, 13]

def load_stanford(filepath, jointCount=19):
    print "Start Loading BVH ... (load_stanford)"
    f = open(filepath, 'r')

    fileNames = []
    jointPosBatch = []

    lines = f.readlines()
    for line in lines:

        trim = line.find('.bmp')

        if trim == -1:
            tpArray = []

            for m in re.finditer(r"\((\d+),\s*(\d+),\s*(\d+)\)", line):
                tpArray.append(int(m.group(1)))
                tpArray.append(int(m.group(2)))
                tpArray.append(int(m.group(3)))

            tpArray = np.reshape(tpArray, (jointCount, 3))
            jointPosBatch.append(tpArray)
        else:
            fileNames.append(line)

    print "BVH Loading Complete!"
    return jointCount, fileNames, jointPosBatch
    #"\d+\.((\w+)\s\(\d+\)\.bmp)\s((\((\d+),\s*( \d+),\s*(\d+)\)){19})"
def load_KB(filepath):
    return False, False, False
    #"\d+\.(([\w_\d\s]+)\.bmp)\s((\((\d+),\s*(\d+),\s*(\d+)\)){19})"

def save_stanford(filepath, fileNames, resultPos):
    fout = open(filepath, 'w')

    for jp, filename in zip(resultPos, fileNames):
        fout.write(filename+' ')
        for coor in jp:
            fout.write('(')
            fout.write(str(int(coor[0])))
            fout.write(', ')
            fout.write(str(int(coor[1])))
            fout.write(', ')
            fout.write(str(int(coor[2])))
            fout.write(')')
        fout.write('\n')

    fout.close()

    return

def save_binary(filepath, resultPos):
    fout = open(filepath, 'w')
    for jp in resultPos:
        for coor in jp:
            for xyz in coor:
                fout.write(str(xyz))
                fout.write(' ')
        fout.write('\n')
    fout.close()

    return
