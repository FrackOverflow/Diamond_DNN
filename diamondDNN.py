import tensorflow as tf
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
import IPython
import numpy as np
import csv
import kerastuner as kt
from tensorflow import keras

####################
# HELPER FUNCTIONS ############################################################
####################

# Reads all rows of CSV into a matrix
# csv_x: csv file name (must be in same dir as this python file)
def readCSV(csv_x):
    readOut = []
    with open(csv_x) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            readOut.append(row)
    return readOut

# Clears duplicate rows by organizing rows into tuples and comparing them
# Returns number of records removed
# mtx: matrix of data
def clearDupes(mtx):
    oldLen = len(mtx)
    newMtx = [tuple(row) for row in mtx]
    mtx = np.unique(mtx, axis=0)
    rmvd = oldLen - len(mtx)
    return rmvd

# Returns a column from a matrix
# mtx: matrix
# i: index
def getCol (mtx, i):
    return [row[i] for row in mtx]

# Sets a column of a matrix to provided values
# mtx: matrix
# i: index
# vals: array of values to set
def setCol(mtx, i, vals):
    j = 0
    while j < len(mtx):
        mtx[j][i] = vals[j]
        j += 1

# Finds illegal values in an array, given a list of legal values
# xs: array to search
# xmap: array of legal values
def findIllegal(xs, xmap):
    i = 0
    illegal = []
    while i < len(xs):
        if not xs[i] in xmap:
            illegal.append(i)
        i += 1
    return illegal

# Removes records from mtx specified by Index
# mtx: matrix to alter
# idxs: indices to remove
def popIllegal(mtx, idxs):
    popped = 0
    while len(idxs) > 0:
        mtx = np.delete(mtx, max(idxs), 0)
        idxs = np.delete(idxs, len(idxs)-1)
        popped += 1
    return popped

# Finds NaN values in an array, first checking if any exist
# xs: array to check
def findNaN(xs):
    illegal = []
    if np.isnan(np.sum(np.array(xs).astype(np.float))):
        i = 0
        while i < len(xs):
            if np.isnan(xs[i].astype(np.float)):
                illegal.append(i)
                i += 1
    return illegal

# Applies min max scaling to an array and returns the scaled array
# xs: array of numerical elements
def minMaxer(xs):
    xmin = min(xs)
    xrng = max(xs) - xmin
    ys = []
    for x in xs:
        ys.append((x - xmin)/xrng)
    return ys

# Applies standard scaling to an array and returns the scaled array
# xs: array of numerical elements
def stdizer(xs):
    xmean = np.mean(xs)
    xstd = np.std(xs)
    ys = []
    for x in xs:
        ys.append((x - xmean)/xstd)
    return ys

# Changes data from categorical to numerical using a feature ranking
# xs: array of categorical elements
# xmap: array of unique values in xs from worst to best
def featureMapper(xs, xmap):
    ys = []
    for x in xs:
        if x in xmap:
            ys.append(xmap.index(x))
        else:
            ys.append(-1)
    return ys

############
# CLEANING ####################################################################
############

### Initialize Data ###
rawData = readCSV("diamonds.csv")                           # Read data from CSV
rawData = np.delete(rawData, 0, 1)                          # Delete Index column
rawData = np.delete(rawData, 0, 0)                          # Remove data headers

### Clean Duplicates ###
popped = clearDupes(rawData)
print("Removed", popped, "Duplicate Records")

### Clean Categorical Columns ###
# Get dirty columns
cut = getCol(rawData, 1)
color = getCol(rawData, 2)
clarity = getCol(rawData, 3)

# Expected values for categorical features
colorMap = ["J","I","H","G","F","E","D"]                                # GIA Color grades
cutMap = ["Fair", "Good", "Very Good", "Premium", "Ideal"]              # GIA Cut grades
clarityMap = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]   # GIA Clarity grades

# Get unique illegal indices, sort in ascending order
dirtyRecords = np.sort(
                    np.unique(
                        [findIllegal(color, colorMap),
                        findIllegal(cut, cutMap),
                        findIllegal(clarity, clarityMap)]
                    )
                )[::-1]

popped = popIllegal(rawData, dirtyRecords)              # Pop illegal records
print("Removed", popped, "records with illegal values") # Result of cleaning

### Clean Numerical Columns ###
carat = getCol(rawData, 0)
depth = getCol(rawData, 4)
table = getCol(rawData, 5)
price = getCol(rawData, 6)
size_x = getCol(rawData, 7)
size_y = getCol(rawData, 8)
size_z = getCol(rawData, 9)
numerical = [carat, depth, table, price, size_x, size_y, size_z]

# Get illegal indices
nanRecords = []
j = 0
while j < len(numerical):
    nanRecords.append(findNaN(numerical[j]))
    j += 1

nanRecords = np.sort(np.unique(nanRecords)[::-1])   # Get unique illegal indices in ascending order
popped = popIllegal(rawData, nanRecords)            # Pop illegal records
print("Removed", popped, "records with NaN values") # Result of cleaning

#################
# PREPROCESSING ###############################################################
#################

### Categorical Preprocessing ###
# Get clean columns
cut = getCol(rawData, 1)
color = getCol(rawData, 2)
clarity = getCol(rawData, 3)

# Map features based on respective ranking map
setCol(rawData, 1, featureMapper(cut, cutMap))
setCol(rawData, 2, featureMapper(color, colorMap))
setCol(rawData, 3, featureMapper(clarity, clarityMap))

### Numerical Preprocessing ###
# Get feature columns
carat = np.array(getCol(rawData, 0)).astype(np.float)
cut = np.array(getCol(rawData, 1)).astype(np.float)
color = np.array(getCol(rawData, 2)).astype(np.float)
clarity = np.array(getCol(rawData, 3)).astype(np.float)
depth = np.array(getCol(rawData, 4)).astype(np.float)
table = np.array(getCol(rawData, 5)).astype(np.float)
size_x = np.array(getCol(rawData, 7)).astype(np.float)
size_y = np.array(getCol(rawData, 8)).astype(np.float)
size_z = np.array(getCol(rawData, 9)).astype(np.float)
numerical = [carat, cut, color, clarity, depth, table, size_x, size_y, size_z]

# Min-Max scale all features
k = 0
colSet = [0, 1, 2, 3, 4, 5, 7, 8, 9]
while k < len(numerical):
    setCol(rawData, colSet[k], minMaxer(numerical[k]))
    k += 1

### Final Preparations ###
np.random.shuffle(rawData) # Randomize record order

# Organize into Train, Validation and Test sets
trainSize = int(np.floor(len(rawData)*0.7))
testSize = int((len(rawData)-trainSize)/2)
trainRaw = rawData[0:trainSize]
valRaw = rawData[trainSize:trainSize+testSize]
testRaw = rawData[trainSize+testSize:]

# Get features for each set
train_x = np.array(np.delete(trainRaw, 6, 1)).astype(np.float)
val_x = np.array(np.delete(valRaw, 6, 1)).astype(np.float)
test_x = np.array(np.delete(testRaw, 6, 1)).astype(np.float)

# Get labels for each set
train_y = np.array(getCol(trainRaw, 6)).astype(np.float)
val_y = np.array(getCol(valRaw, 6)).astype(np.float)
test_y = np.array(getCol(testRaw, 6)).astype(np.float)


############
# ANALYSIS ####################################################################
############

### HYPER PARAMETERS ###
EPOCHS = 100        # Max num of epochs to fit for
tunerEpochs = 40    # Max num of epochs to run on a test model for tuning
searchEpochs = 10   # Number of epochs to search for

### CALLBACKS ###
# Clears Training output during hyper parameter tuning
class ClearTrainOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

# Ends training early when Validation MAPE falls below a threshold
class mapeCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_mean_absolute_percentage_error') < 9.0):
            print("\nValidation MAPE < %9, stopping early")
            self.model.stop_training = True



### Final Model
def buildModel1():
    # Build Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(9, activation=tf.nn.relu, input_shape=(9,)))
    model.add(tf.keras.layers.Dense(36, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(56, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(56, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1))

    # Schedule learning rate
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                    0.01,
                    decay_steps = 37758*10,
                    decay_rate = 1.5,
                    staircase = False)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mape'])
    return model

### 2-Layer Network
def buildModel2():
    # Build Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(9, activation=tf.nn.relu, input_shape=(9,)))
    model.add(tf.keras.layers.Dense(36, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(56, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1))

    # Schedule learning rate
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                    0.01,
                    decay_steps = 37758*10,
                    decay_rate = 1.5,
                    staircase = False)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mape'])
    return model


### 1-Layer Network
def buildModel3():
    # Build Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(9, activation=tf.nn.relu, input_shape=(9,)))
    model.add(tf.keras.layers.Dense(36, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1))

    # Schedule learning rate
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                    0.01,
                    decay_steps = 37758*10,
                    decay_rate = 1.5,
                    staircase = False)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mape'])
    return model

### Less Nodes Model
def buildModel4():
    # Build Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(9, activation=tf.nn.relu, input_shape=(9,)))
    model.add(tf.keras.layers.Dense(12, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(24, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(36, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1))

    # Schedule learning rate
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                    0.01,
                    decay_steps = 37758*10,
                    decay_rate = 1.5,
                    staircase = False)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mape'])
    return model

### MODEL DESIGN ###
def buildModel(hp):
    # Tuning Parameters
    l1_units = hp.Int('unitsL1', min_value = 24, max_value = 44, step = 4)
    l2_units = hp.Int('unitsL2', min_value = 32, max_value = 64, step = 8)
    l3_units = hp.Int('unitsL3', min_value = 40, max_value = 72, step = 8)
    hp_lrn = hp.Choice('learning_rate', values = [0.01, 0.001, 0.0001])
    hp_decay_steps = hp.Choice('decay_steps', values = [10, 100, 1000])
    hp_decay_rate = hp.Choice('decay_rate', values = [0.5, 1.0, 1.5, 2.0])

    # Build Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(9, activation=tf.nn.relu, input_shape=(9,)))
    model.add(tf.keras.layers.Dense(units = l1_units, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units = l2_units, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units = l3_units, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1))

    # Schedule learning rate
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                    hp_lrn,
                    decay_steps = 37758*hp_decay_steps,
                    decay_rate = hp_decay_rate,
                    staircase = False)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mape'])
    return model

### HYPER PARAMETER TUNING ###
# NOTE: After tuning is run trials are saved to "./keras_tuner/diamond_keras",
# to re-tune hyper parameters, this file must be removed

# Create keras hyperband tuner
tuner = kt.Hyperband(buildModel,
                    objective = 'mean_absolute_percentage_error',
                    max_epochs = tunerEpochs,
                    factor = 3,
                    directory = 'keras_tuner',
                    project_name = 'diamond_keras')





# Search parameter space w/ hyperband tuner & return optimal values
tuner.search(train_x, train_y, epochs=searchEpochs, validation_data = (val_x, val_y), callbacks=[ClearTrainOutput()])
tuned_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
#\n\nHyper parameter tuning found the following optimal values
#\n# of nodes in Dense Layer 1: {tuned_hps.get('unitsL1')}
#\n# of nodes in Dense Layer 2: {tuned_hps.get('unitsL2')}
#\n# of nodes in Dense Layer 3: {tuned_hps.get('unitsL3')}
#\nLearning Rate: {tuned_hps.get('learning_rate')}
#\nLearning Rate Time Decay Steps: {tuned_hps.get('decay_steps')}
#\nLearning Rate Time Decay Rate: {tuned_hps.get('decay_rate')}
""")

### MODEL FITTING ###

# Build Model w/ Tuned Hyper Parameters
model = tuner.hypermodel.build(tuned_hps)

# Fit model to test data
history = model.fit(train_x, train_y, epochs = EPOCHS, validation_data = (val_x, val_y), callbacks=[mapeCallback()])

### EVALUATE MODEL ###
print("\n\nResult of Evaluation on Test Data")
results = model.evaluate(test_x, test_y)
print("Test Loss: ", round(results[0], 2))
print("Test MAPE: ", results[1])
