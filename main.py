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
result_dir = 'resultData/result/'

def get_dist(a, b, c, d):
    x = a[0]*d[0] - b[0]*d[0]
    y= a[1]*d[1] - b[1]*d[1]
    z = a[2]*d[2]- b[2]*d[2]
    return math.sqrt(x*x + y*y + z*z)

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
        resultFile = open(resultfilename, 'w')

        total_error = []

        test_set, test_mean, test_std = recon.preprocessModel(test_set, isRefine=False, isNormalize=True)

        for i in range(len(test_set)):

            curr_error = np.zeros((len(self.jointIndex),), dtype=float)

            #create Missing Pos
            for missing_i in range(1, len(self.jointIndex)):
                missingPos, missingMarker = createMissingModel(test_set[i], True, missing_i)

                reconPos = recon.reconstructModel(network, missingPos, missingMarker)

                error_dist = get_dist(test_set[i][missing_i], reconPos[missing_i], test_mean[missing_i], test_std[missing_i])
                curr_error[missing_i] += error_dist

            total_error.append(curr_error)

            if(i % (len(test_set)/100) == 0):
                print i / (len(test_set)/100), "% Complete..."

            #get error

        #print error statistics
        for i in range(1, len(self.jointIndex)):
            error_mean =0.0
            error_std = 0.0
            error_count = 0
            for j in range(0, len(total_error)):
                error_mean += total_error[j][i] / len(total_error)

            print i, "th index Error AVG : ", error_mean
            resultFile.write(str(error_mean)+"\t")

            for j in range(0, len(total_error)):
                error_std += math.sqrt((total_error[j][i] - error_mean) * (total_error[j][i] - error_mean)) / len(total_error)
                if total_error[j][i] > 10.0:
                    error_count += 1

            print i, "th index Error STD : ", error_std
            resultFile.write(str(error_std)+"\t")

            print i, "th index Error Count : ", error_count / float(len(total_error)) * 100 , " %"
            resultFile.write(str(error_count / float(len(total_error)) * 100) + "\n")

            print ""

        resultFile.close()

        return


    def getNewMesh(self, frameCount):
        real_pos = self.test_set[frameCount]
        test_pos = self.test_preprocess_set[frameCount]
        noise_pos, missing = createMissingModel(test_pos, isMissingFixed=True)
        recon_pos = self.network.reconstruct(noise_pos)
        
        for i in range(1,len(test_pos)):

            test_pos[i][0] = test_pos[i][0] * self.test_std[i][0] + self.test_mean[i][0]
            test_pos[i][1] = test_pos[i][1] * self.test_std[i][1] + self.test_mean[i][1]
            test_pos[i][2] = test_pos[i][2] * self.test_std[i][2] + self.test_mean[i][2]

            noise_pos[i][0] = noise_pos[i][0] * self.test_std[i][0] + self.test_mean[i][0]
            noise_pos[i][1] = noise_pos[i][1] * self.test_std[i][1] + self.test_mean[i][1]
            noise_pos[i][2] = noise_pos[i][2] * self.test_std[i][2] + self.test_mean[i][2]

            recon_pos[i][0] = recon_pos[i][0] * self.test_std[i][0] + self.test_mean[i][0]
            recon_pos[i][1] = recon_pos[i][1] * self.test_std[i][1] + self.test_mean[i][1]
            recon_pos[i][2] = recon_pos[i][2] * self.test_std[i][2] + self.test_mean[i][2]

        real_center = copy.copy(real_pos[0])
        test_center = copy.copy(test_pos[0])
        noise_center = copy.copy(noise_pos[0])
        recon_center = copy.copy(recon_pos[0])

        for i in range(len(test_pos)):
            real_pos[i] -= real_center
            if i>0:
                test_pos[i]-= real_center
                noise_pos[i]-= real_center
                recon_pos[i]-= real_center

            real_pos[i]/=100.0
            test_pos[i]/=100.0
            noise_pos[i]/=100.0
            recon_pos[i]/=100.0

        return real_pos, noise_pos, recon_pos
        

    def run_sim(self, network, test_set):
        #self.sim = simulator.Environment(jointIndex, self.getNewMesh)
        test_preprocess_set, test_mean, test_std = recon.preprocessModel(test_set, isRefine=False, isNormalize=True)
        self.test_set = test_set
        self.test_preprocess_set = test_preprocess_set
        self.network=network
        self.test_mean = test_mean
        self.test_std = test_std
        self.sim = simulator.Environment(self.jointIndex, self.getNewMesh)

        while(self.sim()):
            pass


    def train(self, network, training_set, verifying_set, logfilename=None):
        
        pre_set, pre_mean, pre_std = recon.preprocessModel(training_set, isRefine=False, isNormalize=True)
        
        ver_set, ver_mean, ver_std = recon.preprocessModel(verifying_set, isRefine=False, isNormalize=True)

        
        ver_noise_set = []
        for jointPos in ver_set:
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
                        #jointPos = np.reshape(jointPos, (len(self.jointIndex), 3))
                        noisePos, missingMarker = createMissingModel(jointPos, True)
                        noise_batch.append(noisePos)
                    for j in range(self.batchRepetition):
                        current_cost = network.train(batch, noise_batch, miss_rate)

                current_verify_cost, rs = network.verify(ver_set[0:1], ver_noise_set[0:1], miss_rate)

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
            print "NoSuchFileError"
            return
        if ckptname is None:
            curr_date = datetime.now().strftime("%y%m%d%H%M")
            ckptname = curr_date+".ckpt"
            logname= curr_date+".log"
        else:
            print "ckpt already exists!"
            return
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
        training_set, verifying_set, jointNames, jointIndex,  = BVH.load_verify(train_dir)
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
        test_dir = db_dir+"/"+test_filename
        if test_dir==None:
            print "NoSuchFileError"
            return
        #sys. file exists
        if (ckptname is None) or not os.path.exists(ckpt_dir+ckptname):
            print "CKPT Name Required"
            return

        myNN = NeuralNetwork(network_arch=network_arch,
         ckptname = ckpt_dir+ckptname,
         learning_rate=learning_rate,
         decay_rate=decay_rate,
         rectifier=rectifier)
        myNN.load()

        test_set, jointNames, jointIndex= BVH.load(test_dir)

        resultname = ckptname.replace('.ckpt', '.result')

        experiment = Experiment(jointIndex=jointIndex, batchSize=batchSize, batchRepetition=batchRepetition, trainStep=trainStep,
         isMissingFixed=isMissingFixed, missingRate=missingRate, missingCount=missingCount)

        if isSimulation:
            experiment.run_sim(myNN, test_set)
        else:
            experiment.run(myNN, test_set, result_dir+resultname)


if __name__=="__main__":
    argv = sys.argv

    #Noise - Missing Control

    main([15 * 3, 1024, 512, 256],
     db_dir="DB",
     train_filename="jointDB1.bin",
     #test_filename="jointDB2.bin",
     test_filename="jointDB_Stand_Pose_87960.bin",
     ckptname="1612100501.ckpt",
     isTrain=False,
     isMissingFixed=True,
     #rectifier='softplus',
     missingRate=0.3,
     missingCount=1,
     isSimulation=False,
     batchSize=100,
     batchRepetition=5,
     trainStep=300)
