# HCY-Tools
Numpy implementation for converting RGB to HCY and HCY to RGB

## Author
Mathis Petrovich

## Usage
Import the functions
```python3
from hcy import rgb2hcy, hcy2rgb
```

Load an image (pixels have to be between 0 and 1)
```python3
import imageio
img = imageio.imread("lena.png") / 255.
```

Convert the image
```python3
hcy_img = rgb2hcy(img)
oimg = hcy2rgb(hcy_img)
assert np.max(np.abs(oimg - img)) < 10**(-13)
```
