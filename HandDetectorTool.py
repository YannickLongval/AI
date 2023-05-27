"""
Count how many fingers the user is holding up
"""

# import libraries and required classes
import cv2
from cvzone.HandTrackingModule import HandDetector
from langchain.tools import BaseTool
from datetime import datetime
  

class HandDetectorTool(BaseTool):
  name = "Hand Detector"
  description = "use this tool when you need to see how many fingers the user is holding up"

  """
  Once the agent decides to use the HandDetectorTool, the _run function will be called to open the camera, and detect hands for fingers.

  Args:
    uneeded (Any): uneeded variable (required to be compatible with agent)

  Returns an integer that corresponds to the number of fingers the user is holding up.
  """
  def _run(self, uneeded):
    # declaring HandDetector with
    # some basic requirements
    detector = HandDetector(maxHands=1, detectionCon=0.8)

    # keeping track of the start time for the countdown, and initializing the fingersUp variable
    start_time = datetime.now()
    fingersUp = 0
      
    # it detect only one hand from
    # video with 0.8 detection confidence
    video = cv2.VideoCapture(0)

    # keep tracking the jand until 5 seconds have passed
    while((datetime.now() - start_time).seconds <= 6):
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
            fingersUp = sum(detector.fingersUp(lmlist[0]))  
          else:
             fingersUp = 0
        else:
          # if no hand is present, 0 fingers are up
          fingersUp = 0
        # display countdown
        img = cv2.putText(img, str(7 - (datetime.now() - start_time).seconds), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA, False)

        # Display the resulting frame
        cv2.imshow("Video", img)
        cv2.waitKey(1)
    # close camera and return number of fingers shown after 5 seconds have passed.
    video.release()
    cv2.destroyAllWindows()
    return fingersUp

  # function to be run in asynchronous situations, not implemented
  def _arun(self, uneeded):
      raise NotImplementedError("This tool does not support async")