import cv2

img = cv2.imread('ui.jpg')
img_coret = cv2.imread('ui-coret.jpg')

def save(name, frame):
    print(frame)
    cv2.imwrite(name, frame)

save('ui-binthresh.jpg', cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)[1])
save('ui-blur.jpg', cv2.GaussianBlur(img, (51, 51), 0))
save('ui-diff.jpg', cv2.absdiff(img, img_coret))
