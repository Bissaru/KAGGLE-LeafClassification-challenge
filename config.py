import os

TRAIN_CSV_PATH = os.path.join("Data","train.csv")
IMAGES_DIR = os.path.join("Data")
TEST_CSV_PATH = os.path.join("Data","test.csv")
TRAIN_IMAGES_DIR = os.path.join("Data","images")
TEST_IMAGES_DIR = os.path.join("Data","test_images")


IMAGE_SIZE = (32,32)
INPUT_SHAPE = [32,32,1]
VALIDATION_SPLIT = 0.2
SEED = 1
ROTATION_RANGE = 20
FILMODE = "nearest"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
EPOCHS = 50

SHUFFLE_VALUE = True
HORIZONTAL_FLIP = True



