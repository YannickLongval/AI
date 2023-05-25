"""
Count how many fingers the user is holding up
"""

# import libraries and required classes
import cv2
from cvzone.HandTrackingModule import HandDetector

# declaring HandDetector with
# some basic requirements
detector = HandDetector(maxHands=1,
                        detectionCon=0.8)
  
# it detect only one hand from
# video with 0.8 detection confidence
video = cv2.VideoCapture(0)

while True:
      # Capture frame-by-frame 
    _, img = video.read()
    img = cv2.flip(img, 1)
      
    # Find the hand with help of detector
    hand = detector.findHands(img)
      
    if hand:
        
          # Taking the landmarks of hand
        lmlist = hand[0] 
        if lmlist:
            # Find how many fingers are up
            # This function return list
            fingerup = sum(detector.fingersUp(lmlist[0]))  
            img = cv2.putText(img, str(fingerup), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA, False)

    # Display the resulting frame
    cv2.imshow("Video", img)
    cv2.waitKey(1)