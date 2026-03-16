import cv2 as cv
import numpy as np
ref="ogform.png"
sc="scannedform.png"
print("Reading refernce image:",ref)
im1=cv.imread(ref,cv.IMREAD_COLOR)
im1=cv.cvtColor(im1,cv.COLOR_BGR2RGB)
im2=cv.imread(sc,cv.IMREAD_COLOR)
im2=cv.cvtColor(im2,cv.COLOR_BGR2RGB)
im1_gray=cv.cvtColor(im1,cv.COLOR_BGR2GRAY) 
im2_gray=cv.cvtColor(im2,cv.COLOR_BGR2GRAY)
max_features=5000
# oriented fast and rotated brief == it looks for unique points corners and dots
orb=cv.ORB_create(max_features)
kp1,desc1=orb.detectAndCompute(im1_gray, None) #keeps the best one upto ur max_features
kp2, desc2=orb.detectAndCompute(im2_gray,None)#key points are random points aroubd the center of the circle ,descriptors consists of binary if the current pixel is greater than its surrounding then takes it as 1 else takes it as 0
im1_display = cv.drawKeypoints(im1,kp1,outImage=np.array([]),color=(255,0,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
im2_display = cv.drawKeypoints(im2,kp2,outImage=np.array([]),color=(255,0,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
cv.imshow("og_form",im1_display)
cv.waitKey(0)
cv.imshow("scanned_Form",im2_display)
cv.waitKey(0)
matcher=cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)# finds the distance between descriptors 
matches=list(matcher.match(desc1,desc2,None))
matches.sort(key=lambda x: x.distance, reverse=False)
numGoodMatches =int(len(matches)*0.1) # taking top 10 percent of the matchers
matches=matches[:numGoodMatches]
im_matches=cv.drawMatches(im1,kp1,im2,kp2,matches,None)
cv.imshow("og_form",im_matches)
cv.waitKey(0)
pts1=np.zeros((len(matches),2),dtype=np.float32)
pts2=np.zeros((len(matches),2),dtype=np.float32)
for i,match in enumerate(matches):
    pts1[i, :]=kp1[match.queryIdx].pt #since we have stored best features indices in pts ,we are trying to extract co ordinates 
    pts2[i, :]=kp2[match.trainIdx].pt #query index gets the index and .pt tries to extract the co-ordinates

h,mask =cv.findHomography(pts2,pts1,cv.RANSAC) #only coordinates as arguments 
# h (Homography): This is a 3*3 matrix. It is the "formula" that says: "To get from Image 1 to Image 2, rotate by A degrees, scale by B amount, and shift by C pixels.
# "RANSAC: This is a "liar detector." If 10 points say the image moved left, but 1 point says it moved up, RANSAC ignores the "up" point as an outlier.
height,width,channels =im1.shape
im2_reg=cv.warpPerspective(im2, h,(width,height))
cv.imshow("wrap :og",im1)
cv.waitKey(0)
cv.imshow("wrap:scanned ",im2_reg)
cv.waitKey(0)