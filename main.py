import numpy as np
import math as mt
import json as js



def getRotationMatrix(angle):
	omega = angle[0]
	phi = angle[1]
	kappa = angle[2]
	m = np.zeros((3,3))
	m[0,0] = mt.cos(phi) * mt.cos(kappa)
	m[0,1] = mt.sin(omega) * mt.sin(phi) * mt.cos(kappa) + mt.cos(omega)*mt.sin(kappa)
	m[0,2] = -1 * mt.cos(omega) * mt.sin(phi) * mt.cos(kappa) + mt.sin(omega)*mt.sin(kappa)
	m[1,0] = -1 * mt.cos(phi) * mt.sin(kappa)
	m[1,1] = -1 * mt.sin(omega) * mt.sin(phi) * mt.sin(kappa) + mt.cos(omega)*mt.cos(kappa)
	m[1,2] = mt.cos(omega) * mt.sin(phi) * mt.sin(kappa) + mt.sin(omega)*mt.cos(kappa)
	m[2,0] = mt.sin(phi)
	m[2,1] = -1 * mt.sin(omega)*mt.cos(phi)
	m[2,2] = mt.cos(omega)*mt.cos(phi)
	return m

def getMatrixBee(cameraVars, pointVars):
	noOfPoints = len(pointVars)
	B = np.zeros((noOfPoints, 6))
	for i in range(noOfPoints):
		B[i] = np.array(blist(pointVars[i], cameraVars))
	return B

def blist(pointVar, cameraVars):
	pointLeft = pointVar["left"]
	pointRight = pointVar["right"]
	f = cameraVars["focalLength"]
	pp = cameraVars["principalPoint"]

	r1 = np.array([pointLeft[0] - pp[0], pointLeft[1] - pp[1], -1*f])
	r2 = np.array([pointRight[0] - pp[0], pointRight[1] - pp[1], -1*f])
	m1 = cameraVars["rotationMatrix1"]
	m1 = np.transpose(m1)
	m2 = cameraVars["rotationMatrix2"]
	m2= np.transpose(m2)
	b = cameraVars["baseline"]
	a1 = m1 @ r1
	a2 = m2 @ r2

	omega = cameraVars["rightAngles"][0]
	phi = cameraVars["rightAngles"][1]
	kappa = cameraVars["rightAngles"][2]
	
	b1 = -1 * np.linalg.det(np.array([[a1[0], a1[2]], [a2[0], a2[2]]]))

	b2 = np.linalg.det(np.array([[a1[0], a1[1]], [a2[0], a2[1]]]))

	b3 = np.linalg.det(np.array([[b[0], b[1], b[2]], [a1[0], a1[1], a1[2]], [0, -1*a2[2], a2[1]]]))

	diff1 = -1*r2[0]*mt.sin(phi)*mt.cos(kappa) + r2[1]*mt.sin(phi)*mt.sin(kappa) + r2[2]*mt.cos(phi)

	diff2 = r2[0]*mt.sin(omega)*mt.cos(phi)*mt.cos(kappa) - r2[1]*mt.sin(omega)*mt.cos(phi)*mt.sin(kappa)
	diff2 = diff2 + r2[2]*mt.sin(omega)*mt.sin(phi)

	diff3 = -1*r2[0]*mt.cos(omega)*mt.cos(phi)*mt.cos(kappa) + r2[1]*mt.cos(omega)*mt.cos(phi)*mt.sin(kappa) 
	diff3 = diff3 - r2[2]*mt.cos(omega)*mt.sin(phi)

	b4 = np.linalg.det(np.array([[b[0], b[1], b[2]], [a1[0], a1[1], a1[2]], [diff1, diff2, diff3]]))

	diff1 = r2[0]*m2[1,0] - r2[1]*m2[0,0]
	diff2 = r2[0]*m2[1,1] - r2[1]*m2[0,1]
	diff3 = r2[0]*m2[1,2] - r2[1]*m2[0,2]
	b5 = np.linalg.det(np.array([[b[0], b[1], b[2]], [a1[0], a1[1], a1[2]], [diff1, diff2, diff3]]))

	F = -1 * np.linalg.det(np.array([[b[0], b[1], b[2]], [a1[0], a1[1], a1[2]], [a2[0], a2[1], a2[2]]]))

	l = [b1, b2, b3, b4, b5, F]
	return l

def Main():
	f = open("mypoints.json", "r")
	s = f.read()
	pointVars = js.loads(s)
	n = len(pointVars)
	for i in range(n):
		pointVars[i]["left"] = np.array(pointVars[i]["left"])
		pointVars[i]["right"] = np.array(pointVars[i]["right"])
	cameraVars = getInitialEstimates(pointVars)
	cameraVars["rotationMatrix1"] = getRotationMatrix(cameraVars["leftAngles"])
	cameraVars["rotationMatrix2"] = getRotationMatrix(cameraVars["rightAngles"])
	threshold = 0.001
	count = 0
	while(True):
		B = getMatrixBee(cameraVars, pointVars) 
		F = B[:,5]
		if (count > 0) and (np.linalg.norm(F) > oldNorm):
			break
		oldNorm = np.linalg.norm(F)
		B = B[:,0:5]
		delta = np.linalg.lstsq(B, F, rcond=None)[0]
		print("Determinants are")
		print(F)
		makeChanges(cameraVars, delta)
		count = count + 1
		print("After " + str(count) + "th iteration")
		print("The delta vector is:")
		print(delta)
		print("The norm of determs is:")
		print(np.linalg.norm(F))
		print("\n")

	print("angles of the right photo in degrees are")
	print(cameraVars["rightAngles"]*180/mt.pi)
	print("baseline is")
	print(cameraVars["baseline"])


def getInitialEstimates(pointVars):
	cameraVars = {}
	cameraVars["baseline"] = np.array([1, 0, 0])
	cameraVars["baseline"] = cameraVars["baseline"]/(np.linalg.norm(cameraVars["baseline"]))
	omega = 0
	phi = 0
	kappa = 0
	cameraVars["leftAngles"] = np.array([omega, phi, kappa])
	omega = 0.0
	phi = 0
	kappa = 0
	cameraVars["rightAngles"] = np.array([omega, phi, kappa])
	cameraVars["focalLength"] = 152.113
	cameraVars["principalPoint"] = np.array([0,0])
	return cameraVars

	

def makeChanges(cameraVars, delta):
	deltaAngles = delta[2:]
	cameraVars["baseline"] = cameraVars["baseline"] + np.array([0, delta[0], delta[1]])
	cameraVars["rightAngles"] = cameraVars["rightAngles"] + deltaAngles
	cameraVars["rotationMatrix1"] = getRotationMatrix(cameraVars["leftAngles"])
	cameraVars["rotationMatrix2"] = getRotationMatrix(cameraVars["rightAngles"])

if __name__ == '__main__':
	Main()
