# Issues

## contextual.py
gives me an F1 of 98.5 for direct classification(without feeding into image classification). 

names district and address are more easily to get mixed, this is their accuracy:
```
district_incorrect: 36 district_total: 2054 err: 1.75%
name_incorrect: 156 name_total: 2249 error: 6.94%
address_incorrect: 148 address_total: 2116 err: 6.99
state_incorrect: 28 state_total: 2044 err: 1.37%
district_incorrect: 36 district_total: 2054 err: 1.75%
acc 99.34 - f1 98.17
```

## classification.py
uses keras resnet50 and have an accuracy of 88%, will have higher accuracy 98% if i use fastai. 

I manually classify the image in ```pvsh```, but it is difficult to decide whether a image is a printed image or not especially there is a lot of images in the file.
and some printed images are hard to do OCR. should i classify it as printed and feed the result to next layer?

## object_detection.py
for some images, it do not have barcode, need to fix the bug like that.
it also crop too much for some image(3 out of 200), causing lose of information
