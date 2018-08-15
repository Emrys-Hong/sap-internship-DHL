# Issues
## deskew.py
the code i found out online is not good for deskew very blur images, so I uses houghline to detect the lines of the words, i tested on 200 images and 3 images is incorrect.
two because of the address tag layout is different so it is easier to found out lines vertically than horizontally. others are because the line is 180 degree upside down.

I add the white boader when rotating so it easier for tesseract to process. the preprocess for the image is not very suitable for tesseract, can try imagemagick to preprocess the image and then feed it to tesseract, it may have higher accuracy.

## crop.py
sometimes, object detection would give the wrong coordinate to let me crop smaller than the original image. can try to make the crop size bigger than the actual image.

## tesseract.py
needs better preprocessing from ```deskew.py```

