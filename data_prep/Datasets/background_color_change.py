import numpy as np
from PIL import Image

im = Image.open('/home/ssd_storage/datasets/MR/Asian_face_dataset/Asian_face_dataset/asian_hv_mtcnn_all_together/flipped_15_quarter-left.JPG')
# im = im.convert('RGBA')
data = np.array(im)
# just use the rgb values for comparison
rgb = data[:,:,:3]
color = [246, 213, 139]   # Original value
black = [0,0,0, 255]
white = [255,255,255,255]
mask = np.all(rgb == color, axis = -1)
# change all pixels that match color to white
data[mask] = white

# change all pixels that don't match color to black
##data[np.logical_not(mask)] = black
new_im = Image.fromarray(data)
new_im.save('new_file.JPG')