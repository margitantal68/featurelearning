
from enum import Enum

class NormalizationDirection( Enum ):
    ROWS = "rows"
    COLUMNS = "columns"
    USERS = "users-columns"
    NONE = "none"

class NormalizationType( Enum ):
    MINMAX ="minmax"
    ZSCORE ="zscore"
    NONE ="none"

class ModelType( Enum ):
    FCN = "fcn"
    RESNET = "ResNet"
    MLP = "mlp"
    TLENET = "tlenet"
    
class DataType( Enum ):
    GAIT = "gait"
    SIGNATURE = "signature"
    MOUSE = "mouse"

class AugmentationType( Enum ):
    # Circular shift
    CSHIFT ="cshift"
    # Random noise
    RND = "rnd"

DATA_TYPE = DataType.GAIT
MODEL_TYPE = ModelType.FCN

# OUTPUT_FIGURES = "OUTPUT_FIGURES"
TRAINED_MODELS_PATH = "TRAINED_MODELS"
# SAVED_MODELS_PATH = "SAVED_MODELS"

# Init random generator
RANDOM_STATE = 11235

# Model name
# MODEL_NAME = DATA_TYPE.value+"_"+MODEL_TYPE.value+".hdf5"

# Update weights
# UPDATE_WEIGHTS = False

# Use data augmentation
# AUGMENT_DATA = False


# Set verbose mode on/off
VERBOSE = True

# Model parameters
BATCH_SIZE = 16
EPOCHS = 100


# CNN model Input shape GAIT
FEATURES = 128
DIMENSIONS = 3

# Aggregate consecutive segments/blocks
AGGREGATE_BLOCK_NUM = 5
