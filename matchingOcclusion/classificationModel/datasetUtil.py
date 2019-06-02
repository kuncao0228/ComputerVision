# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
from shutil import copyfile
dataDir = "../data/Training/00014/"

mainDataSet = "../data/Training/"



trainDir = "data/Train/"
testDir = "data/Test/"


def createNonStopSigns(dirPath):
    for path in os.listdir(dirPath):
        if (os.path.isdir(mainDataSet + path)) and path != "00014":
            trainRandList = np.random.choice(50,40)
            count = 200;
            for i in trainRandList:
                fileName = os.listdir(mainDataSet + path)[i]
                if(fileName.endswith(".ppm")):
                    print(fileName)
                    copyfile(mainDataSet + path + "/" + fileName, trainDir + str(count) + ".jpg")
                    count += 1
            
    
    for path in os.listdir(dirPath):
        if (os.path.isdir(mainDataSet + path)) and path != "00014":
            testRandList = np.random.choice(50,20)
            testRandList += 50
            count = 100;
            for i in testRandList:
                fileName = os.listdir(mainDataSet + path)[i]
                if(fileName.endswith(".ppm")):
                    print(fileName)
                    copyfile(mainDataSet + path + "/" + fileName, testDir + str(count) + ".jpg")
                    count += 1

    

def createDataSet(dirPath):
    
    trainRandList = np.random.choice(200,200)
    testRandList = np.random.choice(200,100)
    testRandList += 200
    


    count = 0;
    for i in trainRandList:
        fileName = os.listdir(dirPath)[i]
        if(fileName.endswith(".ppm")):
            copyfile(dataDir + fileName, trainDir + str(count) + ".jpg")
            count += 1
            
            
    count = 0;
    for i in testRandList:
        fileName = os.listdir(dirPath)[i]
        if(fileName.endswith(".ppm")):
            copyfile(dataDir + fileName, testDir + str(count) + ".jpg")
            count += 1
            

            
        
        
        

def main():
    createDataSet(dataDir)
    createNonStopSigns(mainDataSet)
    


main()
    
    