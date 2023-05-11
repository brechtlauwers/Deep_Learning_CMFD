import cv2


class SiftExtractor:
    image = None

    def __init__(self, image):
        self.image = image
        self.gray = None

    def extract_features(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(self.gray, None)
        return kp, des

        # cv2.drawKeypoints(self.gray, kp, self.image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imwrite('sift_keypoints.jpg', self.image)
