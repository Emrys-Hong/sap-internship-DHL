from model.object_detection import object_detection
from preprocess.crop import crop
from preprocess.deskew import rotateImg, rotateBarcode
from preprocess.tesseract import tesseract
from model.classification import classify
import fire
from pytesseract import image_to_string
import cv2
import pandas as pd
import math
import re
import string
import numpy as np
import sys
from matplotlib import pyplot as plt
from PIL import Image
from pytesseract import image_to_string
from os import listdir
from os.path import isfile, join
import string
from collections import defaultdict
import re
from collections import Counter
from fuzzywuzzy import fuzz
from sklearn.cluster import KMeans
from skimage.filters import threshold_local
#####################################
# building up thai map dictionary
#####################################

thailand = pd.read_excel('data/thailand_map.xlsx')
eng_thai_province = pd.read_excel('data/eng_thai_province.xlsx')
eng_thai_province.head()
eng_thai_dict = {}
for index in eng_thai_province.index:
    eng_thai_dict[eng_thai_province.loc[index, 'Province (changwat)']] = eng_thai_province.loc[index, 'จังหวัด']
eng_thai_dict['Bangkok'] = 'กรุงเทพมหานคร'
eng_thai_dict['Nongbua Lamphu'] = 'หนองบัวลำภู'
eng_thai_dict['Phangnga'] = 'พังงา'
eng_thai_dict['Prachinburi'] = 'ปราจีนบุรี'
eng_thai_dict['Si Saket'] = 'ศรีสะเกษ'
eng_thai_dict['Singburi'] = 'สิงห์บุรี'
eng_thai_dict['Suphanburi'] = 'สุพรรณบุรี'


df = pd.DataFrame()
province = 'Amnat Charoen'
count = 0
for index in thailand.index:
    df1_dict = thailand.loc[index, 'Province':'Postcode'].to_dict()
    
    if str(df1_dict['Province']) == 'nan':
        df1_dict['province_in_thai'] = eng_thai_dict[province]
        df1_dict['Province'] = province
        if not math.isnan(df1_dict['Postcode']):
            df1 = pd.DataFrame(df1_dict, index=[str(df1_dict['Postcode'])] )
            df = pd.concat([df1,df])
            count += 1
    elif str(df1_dict['Province']) == 'Province':
        pass
    else:
        province = df1_dict['Province']
        df1_dict['province_in_thai'] = eng_thai_dict[province]
        if not math.isnan(df1_dict['Postcode']):
            df1 = pd.DataFrame(df1_dict, index=[ str(df1_dict['Postcode']) ] )
            df = pd.concat([df1, df])
        
        count += 1

df.to_excel('data/thailand_map.xls')
raw_txt = pd.read_excel('data/cleaned_up_raw_txt.xls')
thai_province_list = set(df.loc[:,'province_in_thai'])

# first two digit represent province, except for bangkok.
# build the dictionary for lookup
post2pro = {}
for index in df.index:
    try:
        # bangkok and other province have mixed postcode
        if str(df.at[str(index), 'Postcode'][0])[:2] != '10':
            post2pro[str(df.at[str(index), 'Postcode'][0])[:2]] = df.at[str(index), 'province_in_thai'][0]
        else:
            post2pro[str(df.at[str(index), 'Postcode'][0])] = df.at[str(index), 'province_in_thai'][0]
    except Exception:
        post2pro[str(df.at[str(index), 'Postcode'])[:2]] = df.at[str(index), 'province_in_thai']
        
# add postcode 10280 to dictionary, because it is not in the dataset
post2pro['10280'] = 'สมุทรปราการ'
post2pro['10270'] = 'สมุทรปราการ'
post2pro['10130'] = 'กรุงเทพมหานคร'
post2pro['10200'] = 'กรุงเทพมหานคร'
post2pro['10800'] = 'กรุงเทพมหานคร'
post2pro['10500'] = 'กรุงเทพมหานคร'
post2pro['10560'] = 'กรุงเทพมหานคร'

## building up pro2post
pro2post = {}
for index in df.index:
    try:
        pro2post[ df.at[str(index), 'province_in_thai'][0] ] = str(df.at[str(index), 'Postcode'][0])
    except Exception:
        pro2post[ df.at[str(index), 'province_in_thai'] ] = str(df.at[str(index), 'Postcode'])
pro2post['สมุทรปราการ'] = '10280'
pro2post['กรุงเทพมหานคร'] = '10200'



#####################################
# NER
#####################################

#########
# getting the first 5 numbers as postcode
#########
def get_province_postcode(text, index):
    if len(str(text)) < 10:
        print ('text too short', index)
    postcodes = [str(int(float(s[:5]))) for s in re.findall(r'-?\d+\.?\d*', text) if len(str(int(float(s[:5]))))>=5 and int(float(s[:5])) >= 0]
    province_list = []
    
    for postcode in postcodes:
        try:
            pre_province = df.at[str(postcode),'province_in_thai'][0] if len(df.at[str(postcode),'province_in_thai'][0]) > 1 else df.at[str(postcode),'province_in_thai']
            province_list.append([pre_province,postcode]) # special indexing to get the value
            
            
        except Exception as e:
            try:
                if postcode[:2] != '10':
                    province_list.append([post2pro[postcode[:2]],postcode]) ## deleted [:2] here because might be other except bangkok
                else:
                    province_list.append([post2pro[postcode],postcode])
            except Exception as e:
                pass
    return province_list


#########
# use edit distance for province, since tesseract maybe incorrect
#########
def get_province_levenshtein(text):
   
    words = text.split()[-4:]
    final_dict = {}
    words = [i for i in words if len(i) >= 4]
    dict_score = defaultdict(lambda: 0)
    for word in words:
        # check the correct word that have a lot of common with the word
        purged_word = purge(word)
        # loop through all the states and see which one get higher accuracy       
        

        for province in thai_province_list:
            purged_province = purge(province)
            
            score = fuzz.partial_ratio(purged_word, purged_province)
            dict_score[province] = max(score, dict_score[province])
        
        correct_province = {k:v for k,v in dict_score.items() if v == max(dict_score.values())}
        final_dict = {**final_dict, **correct_province}
    return  list( {k:v for k,v in final_dict.items() if v == max(final_dict.values())}.keys() )
        
            
#########
# purge the word to avoid errors from tesseract
#########
def purge(string):
    new_string = ''
    delete_list = [b'\xe0\xb8\xb1',b'\xe0\xb8\xb3',b'\xe0\xb8\xb4',b'\xe0\xb8\xb5',b'\xe0\xb8\xb6',b'\xe0\xb8\xb7',b'\xe0\xb8\xb8',b'\xe0\xb8\xb9',b'\xe0\xb8\xba',b'\xe0\xb9\x87',b'\xe0\xb9\x88',b'\xe0\xb9\x89',b'\xe0\xb9\x8a',b'\xe0\xb9\x8b',b'\xe0\xb9\x8c',b'\xe0\xb9\x8d',b'\xe0\xb9\x8e']
    for char in string:
        if char.encode('utf') not in delete_list:
            new_string += char
    return new_string

#########
# purged provine from the text after get pre_province
#########
def purge_province(string, province):
    purged_province = purge(province)
    words = string.split()
    for word in words:
        purged_word = purge(word)
        if fuzz.partial_ratio(purged_province, purged_word) == 100:
            string.replace(word, ' ')
            break
    return string

#########
# cleaned_up_list
# to be added
#########
cleaned_up_list = ['Tel', 'ผรบ', 'กรณาสง', 'รหสไปรษณย', 'ชอผรบ', 'tel', 'จ.']

#########
# pre_province and pre_postcode
# used both levenshtein and postcode approach
#########
def get_pro_post(text):
    getPostCode = get_province_postcode(text, index)
    ### result is province

    if  getPostCode == []:

#         print('levenshtein')
        pre_province = get_province_levenshtein(text)


        ## get the postcode through lookup dict
        ### pro2post does not cover all the cases ## fix this

        try:
            pre_postcode = pro2post[pre_province]
        except Exception:
            pre_postcode = 'could not find postcode through province'

        ## purge the postcode, usually not necessary because the postcode is usually not contained there
        try:
            if pro2post[pre_province] in text:
                text = text.replace(pre_postcode, ' ')
        except Exception:
            pass

        ## purge province
        try:
            if pre_province in text:
                text = text.replace(pre_province, ' ')
        except:
            pass

    ### result is postcode    
    elif len(get_province_postcode(text, index)) > 1:
#         print('doueble postcode')
        # chooose which one have a lower levenshtein value compare to the province get in province levenshtein
        # assuming there is gonna be one province levenshtein
        # if we have multiple postcode
        max_score = 0
        postcode = get_province_postcode(text,index)
        leven = get_province_levenshtein(text)

        ## compare and get the one that have max value for input
        for p in postcode:
            for l in leven:
                max_score = max(max_score, fuzz.partial_ratio(p[0], l))
        try:        
            for p in postcode:
                for l in leven:
                    if fuzz.partial_ratio(p[0], l) == max_score:
                        pre_province = p[0]
                        pre_postcode = p[1]
                        raise StopIteration
        except StopIteration:
            pass

        ## purge postcode
        if pre_postcode in text:
            text = text.replace(pre_postcode, ' ')


        ## purge province
        try:
            text = purge_province(text, pre_province)
        except:
            pass

    ### result is postcode
    else:
#         print('postcode')
        try:
            pre_province = get_province_postcode(text,index)[0][0]
            pre_postcode = get_province_postcode(text,index)[0][1]
        except Exception:
            pre_province = 'error'

        ## purge postcode
        try:
            if pre_postcode in text:
                text = text.replace(pre_postcode, ' ')
        except Exception:
            pass

        ## purge province
        try:
            text = purge_province(text, pre_province)
        except:
            pass
    return pre_province, pre_postcode, text

#########
# how long is the digit
# it is used for tel
#########
def digit_len(string):
    count = 0
    for i in string:
        if i.isdigit():
            count += 1
    return count

#########
# have not been used anywhere
#########
def has_digit(string):
    return any(char.isdigit for char in string)

#########
# strip away punctuations
# used in get_name
#########
def get_rid_punctuation(st):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in st if ch not in exclude)
    return s


#########
# name
#########
def get_name(text):
    text_list= text.split('\n')
    name = 'unable to identify name'

    for word in text_list:
        word = get_rid_punctuation(word)
        if len(word.strip()) > 3 and not any(i in purge(word) for i in cleaned_up_list) and digit_len(word) < 3:

            name = word
            break

    text = text.replace(word,' ')
    return name, text

#########
# telphone
#########
def get_tel_address(text):
    text_list = text.split()

    text_num = sorted(text_list,key=digit_len)[-1]

    num = 'unable to identify num'
    if digit_len(text_num) > 8:
        num = text_num

    purged_text = text.replace(num, ' ')
    pre_address = purged_text
    pre_tel = num
    return pre_tel, pre_address

#########
## this function is to purge regular words
## such as 'receipients' name' and 'from' 
#########
def purge_regular(string, cleaned_up_list, name):
    words = string.split()
    for clean in cleaned_up_list:
        for word in words:
            purged_word = purge(word)
            purged_clean = purge(clean)
            if purged_clean in purged_word:
                string = string.replace(word, ' ')
    for word in words:
        if fuzz.ratio('รหัสไปรษณีย์', word)>90:
            string = string.replace(word, ' ')
    for word in string.split('\n'):
        if fuzz.ratio('word', name) > 90:
            string = string.replace(word, ' ')
    return string


#####################################
# ALL IN ONE
#####################################
def combine(txt):

    pre_province, pre_postcode, text = get_pro_post(txt)
    # pre_name
    pre_name, text = get_name(text)
    # pre_tel
    pre_tel, pre_address = get_tel_address(text)
    # purge pre_address
    pre_address = purge_regular(pre_address, cleaned_up_list, pre_name)
    pre_address = pre_address.strip()
    return pre_province, pre_postcode, pre_name, pre_tel, pre_address


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

def test(path_to_image):
    address, bar = to_text(path_to_image)
    print('address', address)
    print('bar', bar)
    if bar:
        print('barcode is:', bar)
    print('sucessfully loaded the model')
    if address: 
        pre_province, pre_postcode, pre_name, pre_tel, pre_address = combine(address)
        print('preprovince', pre_province)
        print('prepostcode', pre_postcode)
        print('prename', pre_name)
        print('pre_tel', pre_tel)
        print('pre_address', pre_address)

if __name__ == '__main__':

    fire.Fire(test)
