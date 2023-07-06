# Import required packages
from pytesseract import Output
import numpy as np
import cv2
import os
import pytesseract


# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = r'..\\TesseractOCR\\tesseract'

# PYTESSERACT_CONFIG = "-c tessedit_char_whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890/-. '"

def raw_OCR(img_src: str):
    filedir = img_src[:-4].replace("Dataset/", "")

    img = cv2.imread(img_src)
    os.makedirs(f"result/raw_{filedir}", exist_ok=True)
    reset_txt_file(f"raw_{filedir}")
    return OCR_to_txt(img, f"raw_{filedir}")


def do_OCR(img_src: str):
    filedir = img_src[:-4].replace("Dataset/", "")
    result = ""
    os.makedirs(f"result/{filedir}", exist_ok=True)

    # Read image from which text needs to be extracted
    img = cv2.imread(img_src)
    cv2.imwrite(f'result/{filedir}/00_img.png', img)

    # Convert the image to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'result/{filedir}/01_gray_img.png', gray_img)

    # Performing OTSU threshold
    ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cv2.imwrite(f'result/{filedir}/02_thresh_img.png', thresh_img)

    # Apply erosion
    eroded_img = erosion(thresh_img)
    cv2.imwrite(f'result/{filedir}/03_eroded_img.png', eroded_img)

    # Apply dilation
    dilated_img = dilation(eroded_img)
    cv2.imwrite(f'result/{filedir}/04_dilated_img.png', dilated_img)

    # Creating a copy of image
    img_copy = img.copy()

    # Finding contours
    contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: get_contour_precedence(x, img.shape[1]))
    cv2.imwrite(f'result/{filedir}/05_contours.png', cv2.drawContours(img, contours, -1, (0, 0, 255), 3))

    reset_txt_file(filedir)

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    counter = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if (w * h < 0.9 * img.shape[0] * img.shape[1]):
            # Cropping the text block for giving input to OCR
            cropped = img_copy[y:y + h, x:x + w]
            cv2.imwrite(f'result/{filedir}/06_contour{counter}.png', cropped)
            text = OCR_to_txt(cropped, filedir)
            result += text + "\n"

        else:
            filled_contour = fill_outside_contour(img_copy, [cnt,])
            # Cropping the text block for giving input to OCR
            filled_cropped = filled_contour[y:y + h, x:x + w]
            cv2.imwrite(f'result/{filedir}/06_contour{counter}.png', img_copy[y:y + h, x:x + w])
            cv2.imwrite(f'result/{filedir}/06_contour{counter}_fill.png', filled_cropped)
            text = OCR_to_txt(filled_cropped, filedir)
            result += text + "\n"
        counter += 1
    return result


def erosion(image, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))):
    return cv2.erode(image, kernel)


def dilation(image, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))):
    return cv2.dilate(image, kernel)


def get_contour_precedence(contour, cols):
    origin = cv2.boundingRect(contour)
    return origin[1] * cols + origin[0]


def fill_outside_contour(image, contour, color=[255, 255, 255]):
    stencil = np.zeros(image.shape).astype(image.dtype)
    cv2.fillPoly(stencil, contour, color)
    return cv2.bitwise_and(image, stencil)


def OCR_to_txt(image, filedir):
    # Open the file in append mode
    file = open(f"result/{filedir}/07_result.txt", "a+", encoding="UTF-8")

    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(image, lang='eng', config="--psm 3")

    # Appending the text into file
    file.write(text)
    file.write("\n")

    # Close the file
    file.close

    return text


def reset_txt_file(filedir):
    file = open(f"result/{filedir}/07_result.txt", "w+", encoding="UTF-8")
    file.write("")
    file.close
