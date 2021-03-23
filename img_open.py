import numpy as np

rows = 512
columns = 512
input_path = "./images/lena.png"
output_path = "./images/lena.jpg"

img_file = open(input_path)
print("Opened file " + input_path)
img = np.fromfile(img_file, dtype = np.uint8, count = rows * columns)

print("Image total size = " + str(img.size))
print("Image array:")
print(img)

img.shape = (img.size//rows, columns)
print("Image matrix:")
print(img)

img.astype('int8').tofile(output_path)
print("Created file " + output_path)