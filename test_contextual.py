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
from fuzzywuzzy import fuzz


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
    words_raw = sentence.strip().split('\n')

    preds = model.predict(words_raw)
    to_print = align_data({"input": words_raw, "output": preds})

    for key, seq in to_print.items():
        model.logger.info(seq)

# create instance of config
config = Config()

# build model
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)



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
    if bar_code_coord:
        bar_code = crop(bar_code_coord, path_to_image)
        deskewd_bar = rotateImg(bar_code)
        bar = image_to_string(deskewd_bar)
        bar = purge_bar(bar)
    else:
        bar = None
        print('unable to locate bar code')
    if address_tag_coord:
        address_tag = crop(address_tag_coord, path_to_image)
        deskewed_address_tag = rotateImg(address_tag)
        cv2.imwrite('tmp/address_tag.jpg', deskewed_address_tag)
        printed = classify('tmp/address_tag.jpg')
        if printed:
            address = tesseract(deskewed_address_tag)
        else:
           address =  None
           print('unable to recognize hand written address tags')
    else:
        address = None
        print('unable to locate address tag')
    return address, bar

def purge(string):
    new_string = ''
    delete_list = [b'\xe0\xb8\xb1',b'\xe0\xb8\xb3',b'\xe0\xb8\xb4',b'\xe0\xb8\xb5',b'\xe0\xb8\xb6',b'\xe0\xb8\xb7',b'\xe0\xb8\xb8',b'\xe0\xb8\xb9',b'\xe0\xb8\xba',b'\xe0\xb9\x87',b'\xe0\xb9\x88',b'\xe0\xb9\x89',b'\xe0\xb9\x8a',b'\xe0\xb9\x8b',b'\xe0\xb9\x8c',b'\xe0\xb9\x8d',b'\xe0\xb9\x8e']
    for char in string:
        if char.encode('utf') not in delete_list:
            new_string += char
    return new_string

def purge_regular(string, cleaned_up_list):
    words = string.split()
    for clean in cleaned_up_list:
        for word in words:
            purged_word = purge(word)
            purged_clean = purge(clean)
            if purged_clean in purged_word:
                string = string.replace(word, ' ')
    for word in words:
        if fuzz.ratio('รหัสไปรษณีย์', word) > 90:
            string = string.replace(word, ' ')
    return string
cleaned_up_list = ['Tel', 'ผรบ', 'กรณาสง', 'รหสไปรษณย', 'ชอผรบ', 'tel', 'จ.']

def test(path_to_image):
    address, bar = to_text(path_to_image)
    print('tesseract output', address)
    address = purge_regular(address, cleaned_up_list)
    if bar:
        print('barcode is:', bar)
    print('sucessfully loaded the model')
    if address: 
        predict_dhl(model, address)

if __name__ == '__main__':

    fire.Fire(test)
