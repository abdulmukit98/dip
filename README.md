# dip

```
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
import skimage.exposure as ex
import skimage.color as color
import scipy.ndimage as ndi
import numpy as np
```


### filter
```
import skimage.data as data

img01 = data.camera()
```

## pip freez
Generate the list of installed packages and extract the output to a text file.

```
pip freeze > requirements.txt
```

Later, on a different machine

```
pip install -r requirements.txt
```

