import fire
from pytesseract import image_to_string




def tesseract(img):
    txt = image_to_string(img, lang='tha+eng')
    return txt


if __name__ == '__main__':
    fire.Fire(tesseract)
