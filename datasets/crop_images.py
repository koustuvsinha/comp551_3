import numpy as np
import scipy.misc # to visualize only
import os
from PIL import Image
import pickle

x = np.fromfile('../Dataset/train_x.bin', dtype='uint8')
x = x.reshape((100000,60,60))

# if you wants to view the image
img = Image.fromarray(x[19])
img.show()

# load the image into a PixelAccess object to address elements
pix = img.load()
print (img.size)            #Get the width and hight of the image for iterating over
print (pix[30,33])          #Get the RGBA Value of a pixel of an image

"""
    A function that takes in the PixelAccess object (which is a 2D array type data-structure) and crops it
    exactly around a bounding-box that contains the two digits and removes redundant surrounding pixels.
"""
def cropImage(pix):
    first__row_white_pixel_index = 60
    first__col_white_pixel_index = 60

    last__row_white_pixel_index = 0
    last__col_white_pixel_index = 0

    # get the first row in which a white pixel appears
    for col in range(0, 60):
        for row in range (0, 60):
            pix_val = pix[row, col]
            if (pix_val == 255 and row < first__row_white_pixel_index):
                first__row_white_pixel_index = row
                break

    # get the first column in which a white pixel occurs
    for row in range(0, 60):
        for col in range (0, 60):
            pix_val = pix[row, col]
            if (pix_val == 255 and col < first__col_white_pixel_index):
                first__col_white_pixel_index = col
                break

    # get the last row in which a white pixel occurs
    for col in range(0, 60):
        for row in range (59, -1, -1):
            pix_val = pix[row, col]
            if (pix_val == 255 and row  > last__row_white_pixel_index):
                last__row_white_pixel_index = row
                break

    # get the last column in which a white pixel occurs
    for row in range(0, 60):
        for col in range (59, -1, -1):
            pix_val = pix[row, col]
            if (pix_val == 255 and col > last__col_white_pixel_index):
                last__col_white_pixel_index = col
                break

    # draw a bounding box according to the above dimensions
    bbox = (first__row_white_pixel_index, first__col_white_pixel_index,
            last__row_white_pixel_index, last__col_white_pixel_index)
    
#     working_slice = img.crop(bbox).save(os.path.join(os.getcwd(), "slice_num" + str(num) + "_.png"))
    return img.crop(bbox)


"""
    function to load data saved in pickel format
"""
def load_data():
    try:
        with open("image.p", "rb") as f:
            x = pickle.load(f)
    except:
        x = []
    return x

"""
    function to save object data in pickel format
"""
def save_data(data):
    with open("image.p", "wb") as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    croppedImages = []          # list that holds all cropped images
    for image in range(0, 100000):
        img = Image.fromarray(x[image])
        pix = img.load()
        croppedImg = cropImage(pix)
        croppedImages.append(croppedImg)

    print (len(croppedImages))  # printing out the length of the list of cropped images to ensure
    save_data(croppedImages)    # save the cropped images into a file

