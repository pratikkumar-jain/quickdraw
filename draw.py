import numpy as np
import cv2
from collections import deque
import pygame
from keras.models import load_model

model = load_model('model_quickdraw.h5')
list_doodle = ['axe', 'bowtie', 'cup', 'diamond', 'envelope', 'fish', 'flower', 'hand', 'lollipop', 'tent']

def playSound(doodle):
	doodle_class = list_doodle[doodle]
	filename = 'voices/'+doodle_class+'.wav'
	pygame.mixer.music.load(filename)
	pygame.mixer.music.play()


lowerBlue = np.array([100, 60, 60])
upperBlue = np.array([140, 255, 255])

kernel = np.ones((5, 5), np.uint8)

points = deque(maxlen=512)

drawboard = np.zeros((471,636,3), dtype=np.uint8)

camera = cv2.VideoCapture(0)
pygame.init()
chance = 0

while(camera.isOpened()):

	retval, image = camera.read()
	image = cv2.flip(image, 1)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	point_mask = cv2.inRange(hsv, lowerBlue, upperBlue)
	point_mask = cv2.erode(point_mask, kernel, iterations=2)
	point_mask = cv2.morphologyEx(point_mask, cv2.MORPH_OPEN, kernel)
	point_mask = cv2.dilate(point_mask, kernel, iterations=1)

	contours, __ = cv2.findContours(
	    point_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
	center = None

	if len(contours) >= 1:
		contour = max(contours, key=cv2.contourArea)
		if cv2.contourArea(contour) > 200:
			((x, y), radius) = cv2.minEnclosingCircle(contour)
			cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(image, center, 5, (0, 0, 255), -1)
			M = cv2.moments(contour)
			center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
			points.appendleft(center)
			for i in range(1, len(points)):
				if points[i - 1] is None or points[i] is None:
					continue
				cv2.line(drawboard, points[i - 1], points[i], (255, 255, 255), 7)
				cv2.line(image, points[i - 1], points[i], (0, 0, 255), 2)
			chance = 1

	elif len(contours)==0:
		if points != []:
			drawboard_grayScale = cv2.cvtColor(drawboard, cv2.COLOR_BGR2GRAY)
			draw_blur = cv2.medianBlur(drawboard_grayScale, 15)
			draw_blur = cv2.GaussianBlur(draw_blur, (5, 5), 0)
			draw_thresh = cv2.threshold(drawboard_grayScale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
			contours = cv2.findContours(draw_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
			if len(contours) >= 1:
				contour = max(contours, key=cv2.contourArea)
				if cv2.contourArea(contour) > 1000:
					x, y, w, h = cv2.boundingRect(contour)
					image = drawboard_grayScale[y:y + h, x:x + w]
					image = cv2.resize(image, (28, 28))
					image = np.array(image, dtype=np.float32)
					image = np.reshape(image, (-1, 28, 28, 1))
					probabilties = model.predict(image)[0]
					doodle = list(probabilties).index(max(probabilties))
					# cv2.imshow(image)
					print(probabilties)
					print(doodle)
					playSound(doodle)
			# pygame.init()
						# if chance == 1:
						# 	playSound('voices/axe.wav')
		# chance = 0
			points = deque(maxlen=512)
			drawboard = np.zeros((471,636,3), dtype=np.uint8)


	cv2.imshow("Paint", drawboard)
	# cv2.imshow("Board", image)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
