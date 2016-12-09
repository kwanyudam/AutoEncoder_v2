import math
import sys
import os
import os.path
from random import randrange


import Reconstructor as recon
import simulator
import BVH
from nn import NeuralNetwork
import numpy as np
import random
import copy
import time
from datetime import datetime

myNN=None
ckpt_dir = 'resultData/ckpt/'
log_dir = 'resultData/log/'

def get_dist(a, b):
    return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]))

def createNoiseModel(jointPos, missRate=0.3):
    noisePos = copy.copy(jointPos)
    for i in range(len(jointPos)):
        if random.random()<missRate:
            noisePos[i][0] = noisePos[i][0] + random.random() - 0.5
            noisePos[i][1] = noisePos[i][1] + random.random() - 0.5
            noisePos[i][2] = noisePos[i][2] + random.random() - 0.5

    return noisePos

def createMissingModel(jointPos, isMissingFixed=False, missingCoor=-1, missRate=0.0):
    missingPos = copy.copy(jointPos)
    missingMarker = [False]*len(jointPos)

    if isMissingFixed:
        i = missingCoor
        if i == -1:
            i = random.randrange(1, len(jointPos))
            missingMarker[i]=True

        missingPos[i][0]=0
        missingPos[i][1]=0
        missingPos[i][2]=0

    else:
        for i in range(len(jointPos)):
            if random.random()<missRate:
                missingPos[i][0]=0
                missingPos[i][1]=0
                missingPos[i][2]=0

    return missingPos, missingMarker

class Experiment:
    def __init__(self, jointIndex, batchSize, batchRepetition, trainStep, isMissingFixed, missingRate, missingCount):
        self.jointIndex = jointIndex
        self.batchSize = batchSize
        self.batchRepetition=batchRepetition
        self.trainStep = trainStep
        self.isMissingFixed = isMissingFixed
        self.missingRate = missingRate
        self.missingCount = missingCount
        return


    def run(self, network, test_set, resultfilename):
        resultFile = open(resultfilename)

        for i in range(len(meshes)):
            jointPos, trans, scale = recon.preprocessModel(test_set[i], False)

            #create Missing Pos

            #get error

        #print Error
        return




    def run_statistics(self, recon, meshes, filename):

        statFile = open(filename.replace('.ckpt', '')+'_statistics.txt', 'w')
        error_sum=0
        error_array = []

        cnt=0
        for index in range(0, len(meshes)):
            originalMesh = meshes[index]
            isUsable, originalMesh, trans, scale = recon.preprocessModel(originalMesh, False)

            tp_error=np.zeros((len(self.jointIndex),), dtype=float)

            for missingIndex in range(1, len(self.jointIndex)):
                missingMesh, missingMarker = createMissingMesh(originalMesh, self.jointIndex, missingIndex)

                reconMesh = recon.reconstructModel(network, missingMesh, missingMarker)

                error_dist = get_dist(originalMesh[missingIndex], reconMesh[missingIndex]) * scale
                tp_error[missingIndex] += get_dist(originalMesh[missingIndex], reconMesh[missingIndex]) * scale


            error_array.append(tp_error)

            if((index)% (len(meshes)/100) == 0):
                print cnt, " % Complete..."
                #if cnt == 3:
                #    break
                cnt = cnt+1
                


        print len(error_array)
        for i in range(0, len(self.jointIndex)):
            error_mean =0.0
            error_std = 0.0
            error_count = 0
            for j in range(0, len(error_array)):
                error_mean += error_array[j][i] / len(error_array)

            print i, "th index Error AVG : ", error_mean
            statFile.write(str(error_mean)+"\t")

            for j in range(0, len(error_array)):
                error_std += math.sqrt((error_array[j][i] - error_mean) * (error_array[j][i] - error_mean)) / len(error_array)
                if error_array[j][i] > 10.0:
                    error_count += 1

            print i, "th index Error STD : ", error_std
            statFile.write(str(error_std)+"\t")

            print i, "th index Error Count : ", error_count / float(len(error_array)) * 100 , " %"
            statFile.write(str(error_count / float(len(error_array)) * 100) + "\n")

            print ""

        statFile.close()

        return
    '''
    def run(self, recon, meshes):

        self.meshes = meshesgd
        self.recon = recon
        hi
        #self.sim = simulator.Environment(jointIndex, self.getNewMesh)
        self.sim = simulator.Environment(self.jointIndex, self.getNewMesh)

        while(self.sim()):
            pass
    '''

    def train(self, network, training_set, verifying_set, logfilename=None):
        
        pre_set = []
        for jointPos in training_set:
            pre_jointPos, trans, scale = recon.preprocessModel(jointPos, isRefine=False, isNormalize=True)
            pre_jointPos = np.reshape(pre_jointPos, (len(self.jointIndex), 3))
            pre_set.append(pre_jointPos)
        ver_set = []
        for jointPos in verifying_set:
            pre_jointPos, trans, scale = recon.preprocessModel(jointPos, isRefine=False, isNormalize=True)
            pre_jointPos = np.reshape(pre_jointPos, (len(self.jointIndex), 3))
            ver_set.append(pre_jointPos)

        ver_noise_set = []
        for jointPos in verifying_set:
            jointPos = np.reshape(jointPos, (len(self.jointIndex), 3))
            noisePos, missingMarker = createMissingModel(jointPos, True)
            ver_noise_set.append(np.reshape(noisePos, (len(self.jointIndex), 3)))

        batches = [pre_set[i:i+self.batchSize] for i in range(0, len(pre_set), self.batchSize)]


        current_cost = current_verify_cost =0

        #np.reshape so dumb.. should pick one
        try:
            logFile = None
            if logfilename is not None:
                logFile = open(logfilename, 'a+')

            if self.isMissingFixed:
                miss_rate = self.missingCount/float(len(self.jointIndex))
            else:
                miss_rate = self.missingRate
            for i in range(self.trainStep):
                print "Training Percentage : ", i / float(self.trainStep) * 100 , "%"
                print "Current Cost : ", current_cost , " Current Verify Cost : ", current_verify_cost

                if logFile is not None:
                    logFile.write(str(i) + "step\t" + str(round(current_cost, 4)) + "\t" + str(round(current_verify_cost, 4)) + "\n")
                for batch in batches:
                    noise_batch = []
                    for jointPos in batch:
                        jointPos = np.reshape(jointPos, (len(self.jointIndex), 3))
                        noisePos, missingMarker = createMissingModel(jointPos, True)
                        noise_batch.append(noisePos)
                    for j in range(self.batchRepetition):
                        #if logFile is not None:
                        #    logFile.write(str(current_cost)+"\t")
                        current_cost = network.train(batch, noise_batch, miss_rate)
                        #print "Current Cost : ", current_cost

                    current_verify_cost = network.verify(ver_set, ver_noise_set, miss_rate)

            if logFile is not None:
                logFile.close()

            network.save()
        except KeyboardInterrupt:
            print "Keyboard Interrupt...!"
            if logFile is not None:
                logFile.close()
            network.save()


def main(network_arch,
    db_dir="",
    train_filename=None,
    test_filename=None,
    isTrain=False,
    ckptname=None,
    decay_rate=0.99,
    rectifier='relu',
    learning_rate=0.001,
    isMissingFixed=None,
    missingRate=None,
    missingCount=0,
    isSimulation=True,
    batchSize=100,
    batchRepetition=10,
    trainStep=100
    ):

    if isTrain:
        
        print "Training Network..."
        train_dir = db_dir+"/"+train_filename
        if train_dir == None:
            print_error("NoSuchFileError")
            return
        if ckptname is None:
            curr_date = datetime.now().strftime("%y%m%d%H%M")
            ckptname = curr_date+".ckpt"
            logname= curr_date+".log"
        else:
            logname = ckptname.replace(".ckpt", ".log")
        if os.path.isfile(ckptname):
            print ckptname, " File Already Exists!"
            return

        startTime = datetime.now()
        print "ckpt name : ", ckptname
        print startTime.strftime("%y-%m-%d %H:%M:%S"), ", Start Training"

        #Create NN
        #-learning rate
        #-decay rate
        #-rectifier
        myNN = NeuralNetwork(network_arch=network_arch,
         ckptname = ckpt_dir+ckptname,
         learning_rate=learning_rate,
         decay_rate=decay_rate,
         rectifier=rectifier)

        #Load BVH
        training_set, verifying_set, jointNames, jointIndex = BVH.load_verify(train_dir)
        training_cnt = len(training_set)
        verifying_cnt = len(verifying_set)

        experiment = Experiment(jointIndex=jointIndex, batchSize=batchSize, batchRepetition=batchRepetition, trainStep=trainStep,
         isMissingFixed=isMissingFixed, missingRate=missingRate, missingCount=missingCount)

        #Train NN
        experiment.train(myNN, training_set, verifying_set, log_dir+logname)

        return

    else:
        print "Test Network..."
        #Load NN

        #Load BVH from test set

        #if isSimulation, load simulator.py
    '''
    if isTrain: #if Training
        train_dir = db_dir+train_filename
        if train_dir == None:
            print_error("NoSuchFileError")
            return

        if ckptname is None:
            #ckptname = 
        if os.path.isfile(ckptname):
            print ckptname, " File Already Exists!"
            return        

        startTime = time.time()

        #jointCount, jointNames, training_set = BVH.load(train_filename)
        trainingJointCount, verifyingJointCount, jointNames, training_set, verifying_set = BVH.load_verify(train_filename)
       
        print "Training count = ", trainingJointCount, " Verifying count = ", verifyingJointCount
 
        #miss_rate = 0.3
        network = NeuralNetwork(network_arch)

        #experiment.train(network, training_set, ckptname, trainstep=trainstep)
        experiment.train_verify(network, training_set, verifying_set, ckptname, trainstep=trainstep)
        endTime = time.time()
        totalTime = endTime-startTime
        print "Training Time : ", str(int(totalTime / 3600)), "Hours ",
        totalTime %= 3600
        print str(int(totalTime / 60)), "Min ",
        totalTime %= 60
        print str(int(totalTime)), "Seconds"
        print "Training Complete!"

        return
    else:
        recon = Reconstructor(saverName, network_arch)


    jointCount, jointNames, test_set = BVH.load(test_filename)

    if isSimulation:
        experiment.run(recon, test_set)
    else:
        experiment.run_statistics(recon, test_set, saverName)
    '''


if __name__=="__main__":
    argv = sys.argv

    #Noise - Missing Control

    main([15 * 3, 1024, 512, 256],
     db_dir="DB",
     train_filename="jointDB1.bin",
     test_filename="jointDB2.bin",
     isTrain=True,
     isMissingFixed=True,
     missingRate=0.3,
     missingCount=1,
     isSimulation=False,
     batchSize=100,
     batchRepetition=1,
     trainStep=10)
