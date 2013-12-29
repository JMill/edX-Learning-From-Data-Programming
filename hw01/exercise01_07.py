'''
Homework 1
Perceptron Learning Algorithm
'''
import matplotlib.pylab as pylab
import random
import numpy
import math



def rndmpt():
    return [random.uniform(-1,1), random.uniform(-1,1)]

def pts2Slope(point0, point1):
    '''
    given a 2 [x,y] coords,
    return slope

    y0 - y1 = m (x0 - x1)
    '''
    x0, y0 = point0
    x1, y1 = point1
    return (y0-y1)/(x0-x1)

def ptAndSlope2Intercept( point, slope):
    '''
    given [x,y] coord and slope,
    return the y-intercept, b.

    y = mx + b
    '''
    x, y = point
    return y - slope * x

def makeLine(point0 = rndmpt(), point1 = rndmpt()):
    '''
    returns description of a line as [slope, intercept]
    '''
    slope = pts2Slope(point0, point1)
    return [slope, ptAndSlope2Intercept( point0, slope )]

def lineOutput(x, line):
    '''
    given an input, x, and a line ([slope, b]), return y.

    y = mx + b
    '''
    return line[0] * x + line[1]

def lineOutputList(xList, line):
    '''
    given a list of inputs, return list of outputs
    '''
    output = []
    for input in xList:
        output.append(lineOutput(input,line))
    return output

def separateXYFromtrainingData(trainingData):
    xyColumns = []
    xCoord = []
    yCoord = []
    for datum in trainingData:
        xCoord.append(datum[0])
        yCoord.append(datum[1])
    xyColumns.append(xCoord)
    xyColumns.append(yCoord)
    return xyColumns

def classifyPoint(datum, target):
    '''
    If point is above line, returns +1.
    Otherwise, returns -1.
    '''
    if datum[1] >= lineOutput( datum[0], target ):
        return 1
    else: return 0-1

def updateWeight(w, misclassedPoint):
    w[0] = 1
    #w[0] += misclassedPoint[2] # c
    w[1] += misclassedPoint[0]*misclassedPoint[2]   # x coord
    w[2] += misclassedPoint[1]*misclassedPoint[2]   # y coord
    return w

def makeWeightLine(w): # 'w' is 'weight'
        # Attempt at fixing division by zero error...
        if w[2] == 0:
            w[2] += 0.0000000001

        m = -w[1]/w[2]
        b = -w[0]/w[2]
        return [m,b]


#### Distance from Point to Line 
#### source: http://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/ 
#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
def distancePointLine (px, py, x1, y1, x2, y2):

    def lineMagnitude (x1, y1, x2, y2):
        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude

    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)
 
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
 
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)
 
    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
        #print DistancePointLine
    return DistancePointLine



def runIteration(trainingData, weight, misclassifiedPts, error, i, color, learned):
    ##
    ### for each PLA iteration...
    ##
    if i == 0: #Make purely random line
        ##### generate random test points to make guess line
        ##### generate hypothesis
        guessPt1 = rndmpt()
        guessPt2 = rndmpt()
        guess = makeLine(guessPt1, guessPt2)
        #pylab.plot( [-1,1], lineOutputList([-1,1], guess), 'y-')
    else: 
        guess = makeWeightLine(weight)
        #print weight

    #### test hypothesis against trainingData
    for datum in trainingData:
        if datum[2] != classifyPoint(datum, guess):
            misclassifiedPts.append(datum)
            #print "Misclassified: ", datum # DEBUG
        if type(color) == type(0.0):
            color = str(color)
        else: pass
        pylab.plot( [-1,1], lineOutputList([-1,1], guess), '-', color=color)
    return misclassifiedPts


def train(trainingData, iterationLimit):
    learned = False
    i = 0
    weight = [0.,0.,0.] # weight vector # w[0] + w[1]x + w[2]y =0

    while learned == False:
        #print weight
        misclassifiedPts = []
        error = 0 # THIS WILL BE USED LATER
        
        ## colors
        if i < iterationLimit-1:
            color = 1-(float(i)/iterationLimit)
            if color > 0.8:
                color = 0.8
        else: color = 'cyan'

        if i == iterationLimit:
            print "Hit iteration limit of ", iterationLimit
            break
        
        misclassifiedPts = runIteration(trainingData, weight, misclassifiedPts, error, i, color, learned)

        if len(misclassifiedPts) == 0:
            learned = True
            print "learned: ", learned, "in", i+1, "iterations"
            break
        else:
            # choose one of the misclassified points by which to adjust the weights
            updateWeight(weight,random.choice(misclassifiedPts))
        i+=1
    #print "here is weight: ", weight
    return i+1, weight


def runTrial(showCharts=False):

    def calcTestDisagreements(iterations, weight):
        disagreements = 0
        numTestPoints = 1000
        testData = []

        for i in range(numTestPoints):
            testData.append(rndmpt())
        for datum in testData:
            if classifyPoint(datum, targetLine) != classifyPoint(datum, makeWeightLine(weight)):
                disagreements += 1

        return disagreements, numTestPoints

    def calcProbDisagreement(disagreements, numTestPoints):
        return float(disagreements)/numTestPoints

    ##### generate random trainingData
    trainingData = [] 
    iterationLimit = 800

    for item in range(numTrainingPoints):
        # generate random x/y coords and
        # set third dimension to 0 
        # (will be -/+1 once run through target function)
        x, y = rndmpt()
        trainingData.append([x, y, 0]) 


    ##### generate target function
    ### choose a line
    # get a pair of points and generate slope/intercept
    targetpoint1 = rndmpt()
    targetpoint2 = rndmpt()
    targetLine= makeLine(targetpoint1, targetpoint2)

    ### set one side to +1 and other to -1
    for datum in trainingData:
        datum[2] = classifyPoint(datum, targetLine)


    iterations, weight = train(trainingData, iterationLimit)
    #print weight
    disagreements, numTestPoints = calcTestDisagreements(iterations, weight)
    trialProbDisgreement = calcProbDisagreement(disagreements, numTestPoints)
    #print trialProbDisgreement

    if showCharts:
        ### PLOTTING ###

        # set axes
        pylab.ylim([-1,1])
        pylab.xlim([-1,1])

        # plot axes guidelines
        #pylab.plot( [-1,1], [0,0], '-', color="0.9")
        #pylab.plot( [0,0], [-1,1], '-', color="0.9")


        # target function
        #pylab.plot(targetpoint1[0], targetpoint1[1], 'bo')
        #pylab.plot(targetpoint2[0], targetpoint2[1], 'bo')
        pylab.plot( [-1,1], lineOutputList([-1,1], targetLine), 'r-')

        # trainingData scatter
        highCoords = []
        lowCoords = []
        for datum in trainingData:
            if datum[2] == 1: highCoords.append(datum)
            else: lowCoords.append(datum)
        pylab.plot(separateXYFromtrainingData(highCoords)[0], separateXYFromtrainingData(highCoords)[1], 'ro')
        pylab.plot(separateXYFromtrainingData(lowCoords)[0], separateXYFromtrainingData(lowCoords)[1], 'co')

        # finish
        pylab.show()

    return trialProbDisgreement, iterations
    #return disagreements, numTestPoints



##### EXPERIMENT #####
trials = 3
random.seed(3)
numTrainingPoints = 100

experimentIterations = []
iterationAverage = 0

experimentProbDisagreements = []
disagreementRate = 0 


for trial in range(trials):
    trialProbDisagreement, trialIterations = runTrial(showCharts=False)

    experimentIterations.append(trialIterations)
    iterationAverage = float(sum(experimentIterations))/len(experimentIterations)

    experimentProbDisagreements.append(trialProbDisagreement)
    disagreementRate = float(sum(experimentProbDisagreements))/len(experimentProbDisagreements)


print iterationAverage
print disagreementRate


