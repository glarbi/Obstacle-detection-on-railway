import numpy as np
from PIL import Image

# Open image and make RGB and HSV versions
RGBim = Image.open("area.jpg").convert('RGB')
HSVim = RGBim.convert('HSV')

# Make numpy versions
RGBna = np.array(RGBim)
HSVna = np.array(HSVim)

# Extract Hue
H = HSVna[:,:,0]

# Find all green pixels, i.e. where 100 < Hue < 140
lo,hi = 100,140
# Rescale to 0-255, rather than 0-360 because we are using uint8
lo = int((lo * 255) / 360)
hi = int((hi * 255) / 360)
green = np.where((H>lo) & (H<hi))
print(green)




