# Importing Image class from PIL module 
from PIL import Image 
 
# Opens a image in RGB mode 
im = Image.open(r"C:\Users\Petru.Sacaleanu\source\repos\MLis-II - The Balcans\machine-learning-in-science-ii-2024\training_data\training_data\284.png") 
 
# Size of the image in pixels (size of original image) 
# (This is not mandatory) 
print(im.size)
 

 

newsize = (120, 120)
im1 = im.resize(newsize)
# Shows the image in image viewer 
im1.show() 