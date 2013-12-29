'''
For Homework 02, Exercieses 01-02. EdX Learning From Data course.
Jonathan Miller
'''

import random

# FUNCTIONS ###########################

def runTrial(numCoins, numFlips):

    def flipCoin():
        if random.random() > 0.5:
            return head
        else:
            return tail


    def findv1(vList):
        return vList[0]

    def findvrand(vList):
        return random.choice(vList)

    def findvmin(vList):
        vmin = 1.
        for v in vList:
            if v < vmin:
                vmin = v
        return vmin


    def sequencesToRatios(flipSequences):
        v1 = 0
        vrand = 0
        vmin =  0
        vList = []

        for sequence in flipSequences:
            numHeads = 0
            #print sequence
            for flip in sequence:
                if flip == head:
                    numHeads += 1.
            vList.append( numHeads / numFlips)
        #print vList
        v1 = findv1(vList)
        vrand = findvrand(vList)
        vmin = findvmin(vList)

        return v1, vrand, vmin


    flipSequences = []
    v1 = 0
    vrand = 0
    vmin = 0
    for coin in range(numCoins):
        coinFlipResults = ""
        for flip in range(numFlips):
            coinFlipResults += flipCoin()
        flipSequences.append(coinFlipResults)

    v1, vrand, vmin = sequencesToRatios(flipSequences)

    return v1, vrand, vmin


# MAIN ###########################



numTrials = 100000
#numTrials = 1
numCoins = 1000
numFlips = 10

v1Exp = 0
vrandExp = 0
vminExp = 0

head = "H"
tail = 't'

for trial in range(numTrials):
    v1Trial, vrandTrial, vminTrial = runTrial(numCoins,numFlips)
    #print v1Trial, vrandTrial, vminTrial
    v1Exp += v1Trial
    vrandExp += vrandTrial
    vminExp += vminTrial

v1Exp /= numTrials
vrandExp /= numTrials
vminExp /= numTrials    

print v1Exp, vrandExp, vminExp
