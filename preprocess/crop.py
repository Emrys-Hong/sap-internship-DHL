import cv2
import fire

def crop(final_list, image_path):
    image = cv2.imread(image_path)
    xmin = int(final_list[0][0])
    ymin = int(final_list[0][1])
    xmax = int(final_list[0][2])
    ymax = int(final_list[0][3])
    crop_img = image[ymin:ymax, xmin:xmax]
    return crop_img
if __name__=='__main__':
    fire.Fire(crop)
