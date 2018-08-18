from model.object_detection import object_detection
from preprocess.crop import crop
from preprocess.deskew import rotateImg, rotateBarcode
from preprocess.tesseract import tesseract
from model.classification import classify
import fire
from pytesseract import image_to_string
import cv2

from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config



def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def predict_dhl(model, sentence):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel
        sentence: sentence from tesseract you want to test

    """
    words_raw = sentence.strip().split(" ")

    preds = model.predict(words_raw)
    to_print = align_data({"input": words_raw, "output": preds})

    for key, seq in to_print.items():
        model.logger.info(seq)


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)
    return model



def lennum(string):
    '''how many number in a string'''
    count = 0
    for c in string:
        if c.isdigit():
            count += 1
    return count

def purge_bar(string):
    ''' purge irrelevant things in OCR'''
    string = string.split()
    for word in string:
        if lennum(word) > 5:
            return word

def to_text(path_to_image):
    address_tag_coord, bar_code_coord = object_detection(path_to_image)
    bar_code = crop(bar_code_coord, path_to_image)
    address_tag = crop(address_tag_coord, path_to_image)
    deskewed_address_tag = rotateImg(address_tag)
    deskewd_bar = rotateImg(bar_code)
    cv2.imwrite('tmp/address_tag.jpg', deskewed_address_tag)
    printed = classify('tmp/address_tag.jpg')
    address = tesseract(deskewed_address_tag) if printed else None
    bar = image_to_string(deskewd_bar)
    bar = purge_bar(bar)
    return address, bar

def test(path_to_image):
    address, bar = to_text(path_to_image)
    print('barcode is:', bar)
    model = main()
    print('sucessfully loaded the model')
    predict_dhl(model, address)

if __name__ == '__main__':
    fire.Fire(test)
