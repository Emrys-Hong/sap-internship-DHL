import cv2
import fire

def crop(final_list, PATH_TO_IMAGE):
    image = cv2.imread(PATH_TO_IMAGE)
    xmin = int(final_list[0][0])
    ymin = int(final_list[0][1])
    xmax = int(final_list[0][2])
    ymax = int(final_list[0][3])
    crop_img = image[ymin:ymax, xmin:xmax]
    return crop_img
if __name__=='__main__':
    fire.Fire(crop)
