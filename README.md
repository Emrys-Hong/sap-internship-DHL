# DHL-pipline-contextual-method
## dependency
install the dependency in requirements.txt
cuda8.0     tensorflow 1.3.0

## Procedure
```model/object_detection.py``` input: raw image from DHL, output: four coordinate for address and bar code

```preprocess/crop.py``` input: coordinate from object detection, output: cropped image for image and bar code

```preprocess/deskew.py``` input: image; output: deskewed image

```preprocess/tesseract.py``` input: image; output: text after OCR

```model/classification.py``` input: image from deskew; output: a binary value whether it is printed

```model/contextual.py``` input: text from tesseract.py; output: structured final data

```test.py``` test the result for pipline.
