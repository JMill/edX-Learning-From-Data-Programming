'''
For Homework 02. EdX Learning From Data course.
Jonathan Miller
'''

import random
import numpy
import numpy.linalg


#  ###########################

def generatePoints(numberOfPoints):
    '''
    Note: Generates Points as well as an target function.

    Returns two points that determine the target function (a line),
    as well as a list of random points and their binary classification 
    (as -1 or +1) based on being below or above the target line.

    format:
    (target x1, target y1, target x2, target y2, [list of classified points as [ 1 (threshold), x coord, y coord, 1 or -1 classification])
    '''
    #generate two points for target function
    x1 = random.uniform( -1, 1 )
    y1 = random.uniform( -1, 1 )
    x2 = random.uniform( -1, 1 )
    y2 = random.uniform( -1, 1 )
    points = []

    #generate random points and classify them based on target function
    for point in range(numberOfPoints):
        x = random.uniform( -1, 1 )
        y = random.uniform( -1, 1 )
        points.append([1, x, y, targetFunctionResult(x1, y1, x2, y2, x, y)]) # ( threshold, xcoord, ycoord, -1/+1 binar output)
    return x1, y1, x2, y2, points


def targetFunctionResult(x1, y1, x2, y2, testx, testy):
    # based on: http://stackoverflow.com/questions/3461453/determine-which-side-of-a-line-a-point-lies
    u = (x2-x1)*(testy-y1) - (y2-y1)*(testx-x1)
    if u >= 0:
        return 1
    else:
        return -1

def linearRegression(points):
    '''
    The data is located in the input list's position 3. Format:
        [ threshold (1), xcoord, ycoord, classification (-1 or +1) ]

    Based on "linear regression algortithm" on Lecture 03 slide 17.

    Returns the weights as a 3-item list.
    '''
    X = []
    y = []
    y_location = len(points[0]) - 1 # locate the binary class value, which is in last position
    for point in points:
        X.append(point[:y_location]) # Add [1, randx, randy ], as numpy array
        y.append(point[y_location]) # add the binary class value (is a -1 or +1)

    #X = numpy.array(X) # conversion not required?
    #y = numpy.array(y) # conversion not required?

    X_psuedo_inverse = numpy.linalg.pinv(X)
    w = numpy.dot( X_psuedo_inverse, y ) # calculate the weights directly.
    return w


############ QUESTION 05

def Ein(weights, points):
    '''

    '''
    errorCount = 0
    y_location = len(points[0]) - 1
    for point in points:
        x = point[:y_location] # x is a vector. [ 1, randx, randy ]
        y = point[y_location] # y is a binary classification. -1 or 1
        if numpy.sign(numpy.dot(weights, x)) != y:
            errorCount += 1.
    return errorCount/len(points)

def runQ05Experiment(numberOfPoints, numberOfTrials):
    '''
    Returns average Ein.
    '''
    Ein_results = []
    for trial in range(numberOfTrials):
        _, _, _, _, points = generatePoints(numberOfPoints) # x1,y1,x2,y2,points
        Ein_results.append(Ein(linearRegression(points),points))
    return numpy.mean(Ein_results)


############ QUESTION 06

def runQ06Experiment(numberOfPointsToTrain, numberOfPointsToTest, numberOfTrials):
    '''
    Note: Slightly bending the rules. We aren't reusing the target function
    or g from Q05. Instead, we'll generate new ones for each trial.

    Returns average Eout.
    '''
    Eout_results = []
    for trial in range(numberOfTrials):
        errorCount = 0
        # make initital target function and training set.
        x1, y1, x2, y2, points = generatePoints(numberOfPointsToTrain)
        # determine hypothesis g.
        weights = linearRegression(points)

        # generate a random point and count the difference in classification
        # between target function and g.
        for point in range(numberOfPointsToTest):
            # makes point of form: [ threshold, randx, randy ]
            point = [ 1, random.uniform(-1,1), random.uniform(-1,1) ]
            # is there a difference in binary classification?
            if numpy.sign(numpy.dot(weights, point[:])) != targetFunctionResult(x1, y1, x2, y2, point[1], point[2]):
                errorCount += 1.
        Eout_results.append(errorCount/numberOfPointsToTest)
    return numpy.mean(Eout_results)



############ QUESTION 07

def perceptron( points, weights = numpy.zeros(3), maxIterations = 1000 ):
    '''
    '''
    isLearned = False
    y_location = len(points[0]) - 1 # Assume that classification (y) is last item
    i = 0

    while not isLearned or (i < maxIterations):
        misclassifiedPoints = []
        for point in points:
            x = point[:y_location] # x is a vector. [ 1, xcoord, ycoord ]
            y = point[y_location] # y is a binary classification. -1 or 1
            # Look for misclassified points based on the current weights.
            if numpy.sign(numpy.dot(weights, x)) != y:
                misclassifiedPoints.append(point)
        if len(misclassifiedPoints) == 0:
            isLearned = True
            break
        else:
            # pick a misclassified point and update the weight vector
            misclassifiedPoint = random.choice(misclassifiedPoints)
            misclass_x = misclassifiedPoint[:y_location]
            misclass_y = misclassifiedPoint[y_location]
            weights = weights + ( numpy.dot(misclass_y, misclass_x) )
        i+=1
    if i >= maxIterations:
        print "Quit after", i, "iterations."
    #print "Learned?", isLearned, "after", i, "iterations."
    return weights, isLearned, i

def runQ07Experiment(numberOfPoints, numberOfTrials):
    '''
    '''
    numberOfIterations = 0.

    for trial in range(numberOfTrials):
        # Find weights with Linear Regression
        _, _, _, _, points = generatePoints(numberOfPoints) # x1,y1,x2,y2,points
        weights = linearRegression(points)
        #print weights

        # Run PLA initialized with LR's weights
        weights, isLearned, iterations = perceptron(points, weights)
        if isLearned:
            numberOfIterations += iterations
    return numberOfIterations / numberOfTrials


############ QUESTION 08

def makeNoiseOnPoints(points, noise):
    '''
    given points and noise as a decimal, return the list 
    '''
    noisyPoints = points[:]
    random.shuffle(noisyPoints)
    numberOfNoisyPoints = int((noise*100)/len(points))
    for number in range(numberOfNoisyPoints):
        noisyPoints[number][3] *= -1 # the y binary classification is reversed
    random.shuffle(noisyPoints)
    return noisyPoints

def targetFunction(x1, x2):
    return numpy.sign( x1**2 + x2**2 - 0.6 )

def generatePointsQ08(numberOfPoints):
    '''
    Note: Generates Points without target function.

    Returns a list of random points and their binary classification 
    (as -1 or +1) based on being below or above the target line.

    format:
    [list of classified points as [ 1 (threshold), x coord, y coord, 1 or -1 classification]
    '''
    points = []

    #generate random points and classify them based on target function
    for point in range(numberOfPoints):
        x1 = random.uniform( -1, 1 )
        x2 = random.uniform( -1, 1 )
        points.append([1, x1, x2, targetFunction(x1, x2)]) # ( threshold, xcoord, ycoord, -1/+1 binar output)
    return points


def runQ08Experiment(numberOfPoints, numberOfTrials, noise):
    '''
    Returns average Ein.
    '''
    Ein_results = []
    for trial in range(numberOfTrials):
        points = makeNoiseOnPoints( generatePointsQ08(numberOfPoints), noise )
        #print points
        weights = linearRegression(points)
        Ein_results.append( Ein(weights,points) )
    return numpy.mean(Ein_results)


############ Question 09

def transformation(point):
    '''
    Expects:
        [ 1, xcoord, ycoord, class ]
    Returns:
        [ 1, x1, x2, x1x2, x1^2, x2^2, classification ]
    '''
    return [ point[0], point[1], point[2], point[1] * point[2], point[1]**2, point[2]**2, point[3] ]

def transformPoints(points):
    i = 0
    for point in points:
        points[i] = transformation(point)
        i+=1
    return points

def runQ09Experiment(numberOfPoints, noise):
    # possible solutions (evaluated with sign())
    g_choices = {
        'a': [ -1, -0.05, 0.08, 0.13, 1.5, 1.5 ],
        'b': [ -1, -0.05, 0.08, 0.13, 1.5, 15 ],
        'c': [ -1, -0.05, 0.08, 0.13, 15, 1.5 ],
        'd': [ -1, -1.5, 0.08, 0.13, 0.05, 0.05 ],
        'e': [ -1, -0.05, 0.08, 1.5, 0.15, 0.15 ]
        }

    # 1. make points. 2. make noise. 3. transform nonlinear points.
    points = transformPoints( makeNoiseOnPoints( generatePointsQ08(numberOfPoints), noise ) )
    weights = linearRegression(points)
    for key, value in g_choices.items():
        errors = 0
        value = numpy.array(value)
        for point in points:
            if numpy.sign(numpy.dot(weights, point[:6])) != numpy.sign(numpy.dot(value, point[:6])):
                errors += 1
        print key, 'has', errors, 'errors'
    return 



############ Question 10

def runQ10Experiment(numberOfPoints, numberOfPointsToTest, numberOfTrials, noise):
    '''
    '''
    Eout_results = []
    for trial in range(numberOfTrials):
        errorCount = 0

        # 1. make points. 2. make noise. 3. transform nonlinear points.
        points = transformPoints( makeNoiseOnPoints( generatePointsQ08(numberOfPoints), noise ) )
        weights = linearRegression(points)

        '''# make out of sample noisy test data
        outOfSamplePoints = transformPoints( makeNoiseOnPoints( generatePointsQ08( numberOfPointsToTest), noise))
        for point in outOfSamplePoints:
            if numpy.sign(numpy.dot(weights, point[:6])) != point[6]:
                errorCount += 1
        Eout_results.append(errorCount/numberOfPointsToTest)
        '''
        # generate a random point and count the difference in classification
        # between target function and g.
        for point in range(numberOfPointsToTest):
            # make a noisy point and transform it.
            x1 = random.uniform(-1,1)
            x2 = random.uniform(-1,1)
            y = targetFunction(x1, x2)
            if random.uniform(0,1) < noise:
                y *= -1
            point = transformation ( [1, x1, x2, y] )

            # is there a difference in binary classification?
            if numpy.sign(numpy.dot(weights, point[:6])) != y:
                errorCount += 1.
        Eout_results.append(errorCount/numberOfPointsToTest)
    return numpy.mean(Eout_results)

############ MAIN
#random.seed(0)

# usage: runQ05Experiment(numberOfPoints, numberOfTrials)
#print runQ05Experiment(100,1000)

# usage: runQ06Experiment(numberOfPointsToTrain, numberOfPointsToTest, numberOfTrials)
#print runQ06Experiment(100, 1000, 1000)

# usage: runQ07Experiment(numberOfPoints, numberOfTrials)
#print runQ07Experiment(10,1000), "average iterations with LR then PLA."

# usage: runQ08Experiment(numberOfPoints, numberOfTrials, noise)
#print runQ08Experiment(1000,1000, 0.1), "average E_in"

# usage: runQ09Experiment(numberOfPoints, noise)
#runQ09Experiment(1000, 0.1)

# usage: runQ10Experiment(numberOfPoints, numberOfPointsToTest, numberOfTrials, noise)
print runQ10Experiment(1000, 1000, 1000, 0.1), 'average E_out'
