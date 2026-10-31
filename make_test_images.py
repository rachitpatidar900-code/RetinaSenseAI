import cv2

img = cv2.imread("sample_images/aptos_000c1434d8d7.png")
cv2.imwrite("blurred_test.png", cv2.GaussianBlur(img, (51, 51), 0))
cv2.imwrite("dark_test.png", cv2.convertScaleAbs(img, alpha=0.2, beta=0))
print("saved blurred_test.png and dark_test.png")