from model.object_detection import object_detection
from preprocess.crop import crop
from preprocess.deskew import rotateImg, rotateBarcode
from preprocess.tesseract import tesseract
from model.classification import classify
from model.contextual import  predict_dhl
import fire

def to_text(path_to_image):
    address_tag_coord, bar_code_coord = object_detection(path_to_image)
    bar_code = crop(bar_code_coord)
    address_tag = crop(address_tag_coord)
    deskewed_address_tag = rotateImg(address_tag)
    deskewd_bar = rotateBarcode(bar_code)
    printed = classify(deskewed_address_tag)
    address = tesseract(deskewed_address_tag) if printed else None
    bar = tesseract(deskewd_bar)
    return address, bar

def test(path_to_image):
    address, bar = to_text(path_to_image)
    print('barcode is:', bar)
    predict_dhl(address)

if __name__ == '__main__':
    fire.Fire(test)
