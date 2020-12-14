import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
import PIL.Image as PIL

ThresholdNum = 10

'''
calculate the average mean_iou score of one batch
input: CUDA Variables & batch number
'''
def metric(prediction, target, batch_idx):
	#print('batch %d' % (batch_idx))
	# convert Variable to ndarray
	'''
	prediction = prediction.cpu()
	target = target.cpu()
	prediction = prediction.data.numpy()
	prediction = prediction[0] <= prediction[1]
	target = target.data.numpy()
	'''
	prediction = prediction.cpu().numpy()
	target = target.cpu().numpy()

	score = 0
	for i in range(0, prediction.shape[0]):  # traverse the pictures in a batch
		iou = calculate_iou(prediction[i], target[i], i)
		score_adder = 0
		for t in np.arange(0.5, 1.0, 0.05):
			TP, FP, FN = precision_at(iou, t)
			score_adder += float(TP / (TP + FP + FN))
		score += score_adder / ThresholdNum
	return score, prediction.shape[0]

'''
return a 2D array includes iou between all objects
input: ndarrays & picture index
'''
def calculate_iou(prediction, target, idx):
	# label the prediction and the mask
	prediction = label(prediction, neighbors=8)
	target = label(target, neighbors=8)
	'''
	fig = plt.figure()
	pic_window = fig.add_subplot(121)
	pic_window.imshow(prediction)
	pic_window = fig.add_subplot(122)
	pic_window.imshow(target)
	plt.show()
	'''
	pred_nucs = len(np.unique(prediction))
	targ_nucs = len(np.unique(target))
	'''
	print('pred_num: %d' % (pred_nucs))
	print('targ_num: %d' % (targ_nucs))
	'''

	intersection = np.histogram2d(target.flatten(), prediction.flatten(), bins=(targ_nucs, pred_nucs))[0]
	
	area_targ = np.histogram(target, bins=targ_nucs)[0]
	area_pred = np.histogram(prediction, bins=pred_nucs)[0]
	area_targ = np.expand_dims(area_targ, -1)
	area_pred = np.expand_dims(area_pred, 0)
	union = area_targ + area_pred - intersection

	# exclude background from the ndarray
	intersection = intersection[1:, 1:]
	union = union[1:, 1:]
	union[union==0] = 1e-9

	iou = intersection / union
	'''
	he = 0
	l = 0
	for i in range(0, iou.shape[0]):
		for j in range(0, iou.shape[1]):
			if iou[i][j] >= 0.5:
				he += 1
			elif iou[i][j] > 0:
				l += 1
	print('larger equal: %d' % (he))   # 3
	print('lower than: %d' % (l))	   # 98
	'''
	return iou

'''
returns the TP, FP, FN in terms of precision threshold t
'''
def precision_at(iou, threshold):
	matches = iou > threshold
	true_pos = np.sum(matches, axis=1) == 1
	false_pos = np.sum(matches, axis=0) == 0
	false_neg = np.sum(matches, axis= 1) == 0
	TP, FP, FN = np.sum(true_pos), np.sum(false_pos), np.sum(false_neg)
	return TP, FP, FN
