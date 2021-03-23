import numpy as np

ROWS = 512
COLUMNS = 512
INPUT_PATH = "./images/lena.png"
OUTPUT_PATH = "./images/lena.jpg"

img_file = open(INPUT_PATH)
print("Opened file " + INPUT_PATH)
img = np.fromfile(img_file, dtype = np.uint8, count = ROWS * COLUMNS)

print("Image total size = " + str(img.size))
print("Image array:")
print(img)

img.shape = (img.size//ROWS, COLUMNS)
print("Image matrix:")
print(img)

img.astype('int8').tofile(OUTPUT_PATH)
print("Created file " + OUTPUT_PATH)
