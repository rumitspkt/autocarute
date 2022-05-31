import cv2

stop_sign = cv2.CascadeClassifier("/home/pi/Desktop/autocarute/traffic_sign/stopsign_classifier.xml")
turn_right = cv2.CascadeClassifier("/home/pi/Desktop/autocarute/traffic_sign/turnRight_ahead.xml")
turn_left = cv2.CascadeClassifier("/home/pi/Desktop/autocarute/traffic_sign/turnLeft_ahead.xml")


class TrafficDetection(object):

    # 0 - unknown, 1 - left, 2 - right, 3 - stop
    def signDetected(self, image, debug=False, thresholdRatio=0.19):
        heightFrame, widthFrame = image.shape[:2]
        frameArea = heightFrame * widthFrame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Stop = stop_sign.detectMultiScale(gray, 1.02, 10)
        Turn_Right = turn_right.detectMultiScale(gray, 1.02, 10)
        Turn_Left = turn_left.detectMultiScale(gray, 1.02, 10)

        biggestStopArea = 0
        biggestLeftArea = 0
        biggestRightArea = 0
        stopBox = (0, 0, 0, 0)
        rightBox = (0, 0, 0, 0)
        leftBox = (0, 0, 0, 0)
        for (x, y, w, h) in Stop:
            if w * h > biggestStopArea:
                biggestStopArea = w * h
                stopBox = (x, y, w, h)

        for (x, y, w, h) in Turn_Right:
            if w * h > biggestRightArea:
                biggestRightArea = w * h
                rightBox = (x, y, w, h)

        for (x, y, w, h) in Turn_Left:
            if w * h > biggestLeftArea:
                biggestLeftArea = w * h
                leftBox = (x, y, w, h)

        stopRatio = biggestStopArea / frameArea
        leftRatio = biggestLeftArea / frameArea
        rightRatio = biggestRightArea / frameArea

        box = (0, 0, 0, 0)
        type = 0
        if biggestStopArea > biggestLeftArea and biggestStopArea > biggestRightArea:
            print('ratio stop: {}'.format(stopRatio))
            if stopRatio >= thresholdRatio:
                box = stopBox
                type = 3

        if biggestLeftArea > biggestStopArea and biggestLeftArea > biggestRightArea:
            print('ratio left: {}'.format(leftRatio))
            if leftRatio >= thresholdRatio:
                box = leftBox
                type = 1

        if biggestRightArea > biggestStopArea and biggestRightArea > biggestLeftArea:
            print('ratio right: {}'.format(rightRatio))
            if rightRatio >= thresholdRatio:
                box = rightBox
                type = 2

        if debug:
            if box[0] != 0 and box[1] != 0 and box[2] != 0 and box[3] != 0:
                cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0),
                              2)

        return type
