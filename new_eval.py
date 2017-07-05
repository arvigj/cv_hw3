import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess
import itertools
from multiprocessing import Pool

f = subprocess.check_output(["ls"]).split()
files = []
#make list of files that contain ellipse data
for i in f:
    if "ellipseList.txt" in i:
        files.append(i)
print files

class Image:
	def __init__(self, filename, window_size):
		self.im = cv2.imread(filename,0)
		#self.im = cv2.resize(self.im,(0,0),fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
		self.mask = []
		self.mask_small = []
		self.windows = []
		self.windows_small = []
		self.scores = []
		self.scores_small = []
		self.cx = []
		self.cy = []
		self.decimation_factor = []
		self.imno = 0
		#self.slide = [-6,-4,-2,0,2,4,6]
		self.slide = [-2,-1,0,1,2]
		self.window_size = window_size

	def ellipse(self, ellipse_info):
		ellipse_info = ellipse_info.split(" ")
		axes = [float(ellipse_info[0]),float(ellipse_info[1])]
		decim_fac = int(max(max(axes[0]*2/self.window_size,axes[1]*2/self.window_size),1))
		self.decimation_factor.append(decim_fac)

		#print "best decimation is %.2f and %.2f"%(axes[0]*2/32,axes[1]*2/32)
		theta = float(ellipse_info[2])
		self.cx.append(float(ellipse_info[3]))
		self.cy.append(float(ellipse_info[4]))
		#print "diameter is %0.2f"%(2*max(axes[0],axes[1]))
		y,x = np.ogrid[0:self.im.shape[0],0:self.im.shape[1]]
		mask = np.power(((x-self.cx[-1])*np.cos(theta) + (y-self.cy[-1])*np.sin(theta))/axes[0],2) + np.power(((x-self.cx[-1])*np.sin(theta) - (y-self.cy[-1])*np.cos(theta))/axes[1],2) <= 1
		self.mask.append(mask)
		#self.mask.append(mask[::2,::2])
		#self.cx[-1] /= 2
		#self.cy[-1] /= 2

	def ellipse_decim(self, ellipse_info):
		ellipse_info = ellipse_info.split(" ")
		axes = [float(ellipse_info[0])/2,float(ellipse_info[1])/2]
		print "best decimation is %.2f and %.2f"%(axes[0]*2/32,axes[1]*2/32)
		theta = float(ellipse_info[2])
		self.cx.append(float(ellipse_info[3])/2)
		self.cy.append(float(ellipse_info[4])/2)
		#print "diameter is %0.2f"%(2*max(axes[0],axes[1]))
		y,x = np.ogrid[0:self.im.shape[0],0:self.im.shape[1]]
		mask = np.power(((x-self.cx[-1])*np.cos(theta) + (y-self.cy[-1])*np.sin(theta))/axes[0],2) + np.power(((x-self.cx[-1])*np.sin(theta) - (y-self.cy[-1])*np.cos(theta))/axes[1],2) <= 1
		self.mask.append(mask)


	def get_score(self,mask,cx,cy,x,i,ellipse_size):
		s = self.window_size/2
		flag = False
		flag = flag or cy+x[0]-s < 0
		flag = flag or cx+x[0]-s < 0
		flag = flag or cy+x[1]+s+1 > mask.shape[0]
		flag = flag or cx+x[1]+s+1 > mask.shape[1]
		if flag == True:
			return -1.
		#intersect = np.sum(self.mask[i][cy+x[0]-16:cy+x[0]+17,cx+x[1]-16:cx+x[1]+17]).astype(float)
		#union = ellipse_size - intersect + (32*32)

		intersect = np.sum(mask[cy+x[0]-s:cy+x[0]+s+1,cx+x[1]-s:cx+x[1]+s+1]).astype(float)
		union = ellipse_size - intersect + (4*s*s)
		self.imno += 1

		#CHOOSE THE SCORE YOU WANT
		#return intersect/union
		return intersect/ellipse_size

	def get_random_window(self,image,mask,center):
		s = self.window_size/2
		rand_mask = mask[center[0]-s:center[0]+s+1,center[1]-s:center[1]+s+1]
		if rand_mask.size < (self.window_size**2) or np.sum(rand_mask) > 5:
			return None
		return image[center[0]-s:center[0]+s+1,center[1]-s:center[1]+s+1]

	def get_windows(self):
		s = self.window_size/2
		self.image_slides = []
		self.score_slides = []
		for i in xrange(len(self.mask)):
			image = cv2.resize(self.im,(0,0),fx=1./self.decimation_factor[i],fy=1./self.decimation_factor[i],interpolation=cv2.INTER_AREA)
			mask = cv2.resize(self.mask[i].astype(np.uint8),(0,0),fx=1./self.decimation_factor[i],fy=1./self.decimation_factor[i],interpolation=cv2.INTER_AREA).astype(bool)
			mask_size = np.sum(mask)
			cx = int(round(self.cx[i]/self.decimation_factor[i]))
			cy = int(round(self.cy[i]/self.decimation_factor[i]))
			self.score_slides.append(map(lambda x: self.get_score(mask,cx,cy,x,i,mask_size), itertools.product(self.slide,self.slide)))
			self.image_slides.append(map(lambda x: image[cy+x[0]-s:cy+x[0]+s+1,cx+x[1]-s:cx+x[1]+s+1], itertools.product(self.slide,self.slide)))
		
		#generate random images
		rand = np.random.rand([2*self.imno,2])
		rand[:,0] *= self.im.shape[0]
		rand[:,1] *= self.im.shape[1]
		rand = rand.astype(int)
		x = 0
		y = 0
		map(lambda x: self.get_random_window(self.im,))



		
	def get_data(self):
		return self.image_slides, self.score_slides



def info(filename):
	with open(filename,"r") as f:
		slides = []
		scores = []
		while(True):
			try:
				imgpath = f.readline().split("\n")[0]+".jpg"
				if imgpath == ".jpg":
					return np.array(slides), np.array(scores)
				#print imgpath
				e = Image(imgpath,40)
				numfaces = f.readline().strip()
				#print numfaces
				print numfaces
				for i in xrange(int(numfaces)):
					ellipse_info = f.readline().split("\n")[0]
					#print ellipse_info
					e.ellipse(ellipse_info)
					#plt.imshow(e.im,cmap="gray",alpha=0.5)
					#plt.imshow(e.ellipse(ellipse_info),alpha=0.1,cmap="gray")
					#plt.show()
				e.get_windows()
				ims, im_scores = e.get_data()
				for i in xrange(len(ims)):
					slides.append(ims[i])
					scores.append(im_scores[i])
				#print
				#e.get_windows()
			except ValueError as a:
				#pass
				#    print e
				return
	#return

#info(files[0])
#exit()

pool = Pool(4)
a = np.array(pool.map(info,files))

images = np.concatenate(a[:,0])
scores = np.concatenate(a[:,1])

images = images.ravel()[np.where(scores.ravel() >= 0)]
scores = scores.ravel()[np.where(scores.ravel() >= 0)]

plt.hist(scores,bins=50)
plt.show()

rand_range = (np.random.rand(10)*40000).astype(int)
for i in xrange(10):
	plt.imshow(images[rand_range[i]],cmap="gray",interpolation="nearest")
	print scores[rand_range[i]]
	plt.show()