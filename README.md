# DHL-pipline-contextual-method
## dependency
install the dependency in requirements.txt
cuda8.0     tensorflow 1.3.0

## download trained files
```frozen_inference_graph.pb``` and ```labelmap.pbtxt``` are needed for object_detection.py, can be found in ```trained_models``` folder

classification models and dataset ```pvsh``` and ```4_kind_pre``` can be found in ```trained_models``` folder. i used dataset ```pvsh``` to classify and got an accuracy of 88. need to specify the path in ```classification.py```

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

## download files
download the ```results```(trained models in it) and ```data``` (trained data in it) folder and put it in root directory

## results
for entity_linking code run:
```CUDA_VISIBLE_DEVIECS=1 python test_entity_linking.py test_images/62.png```
predictions:
```
barcode is: TDPSHO720171
preprovince ขอนแก่น
prepostcode 40000
prename อเมืองจขชอนนคน
pre_address ส.จุฑามาศ. เลิน ห้อง 610
หอสทธิลักษณ์ 473/โหมุ27 ด.ติลฯ
```
results:
```
Barcode: TDPSH07201711223
Province: ขอนแก่น
Zipcode: 40000
Name: จุฑามาศ เมลิน ห้อง610
State: ขอนแก่น
Address: หอสุทธิลักษณ์ 473/1 ม.27 ตำบลศิลา
```


for contextual NER code run:
```CUDA_VISIBLE_DEVIECS=1 python test_contextual.py test_images/62.png```
