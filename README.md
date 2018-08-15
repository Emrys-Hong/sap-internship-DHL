# DHL-pipline-contextual-method
## dependency
install the dependency in requirements.txt
cuda8.0     tensorflow 1.3.0

## download trained files
```frozen_inference_graph.pb``` and ```labelmap.pbtxt``` are needed for object_detection.py, can be found in ```trained_models``` folder
```embeddings etc``` are needed in contextual.py, can be found in trained models. download the ```sequence_tagging_thailand``` from the ```trained_model``` folder and put ```model/contextual.py``` in ```sequence_tagging_thailand``` to predict.


## Procedure
```model/object_detection.py``` input: raw image from DHL, output: four coordinate for address and bar code

```preprocess/crop.py``` input: coordinate from object detection, output: cropped image for image and bar code

```preprocess/deskew.py``` input: image; output: deskewed image

```preprocess/tesseract.py``` input: image; output: text after OCR

```model/classification.py``` input: image from deskew; output: a binary value whether it is printed

```model/contextual.py``` input: text from tesseract.py; output: structured final data

```test.py``` test the result for pipline.


## how to generate those files if they do not work
```model/contextual.py``` is using [this github](https://github.com/guillaumegenthial/sequence_tagging) to produce result. they produce SOTA result on coNLL classification tasks. i downloaded the fasttext embedding for thai (if you want to test on english, i used glove embedding. to produce the test data, use ```parcel_data.xls``` and ```generate_contextual_data.ipynb``` in ```extra_file``` folder

## problems and evaluation
```contextual.py``` 
I have tested the contextual.py on the following cases
