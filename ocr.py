import cv2
import pytesseract
cap = cv2.VideoCapture(0)

while True:
    c, img = cap.read()
    cv2.imshow("Test Detection", img)

    # Extract text from the image
    text = pytesseract.image_to_string(img)

    # Print the text
    print(text)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break
cap.release()
cv2.destroyAllWindow()
