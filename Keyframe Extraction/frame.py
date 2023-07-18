import cv2

img = cv2.imread('test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)))
cv2.imshow("Test", closed)
cv2.waitKey(0)
contours, _ = cv2.findContours(
    closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img2 = img.copy()
maxContour = max(contours, key=cv2.contourArea)
# for cnt in contours:
x, y, w, h = cv2.boundingRect(maxContour)
img2 = img2[y:y+h, x:x+w]
cv2.imshow("Shapes", img2)
cv2.waitKey(0)
