from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.visualize import save_image
import skimage.color
import skimage.io
import skimage.transform
 
class NewClassDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "new_class")
		# define data locations
		if is_train:
			images_dir = dataset_dir + '/train/images/'
			annotations_dir = dataset_dir + '/train/annots/'
		else:
			images_dir = dataset_dir + '/test/images/'
			annotations_dir = dataset_dir + '/test/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 
	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
 
	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('new_class'))
		return masks, asarray(class_ids, dtype='int32')
 
	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
 
# define a configuration for the model
class NewClassConfig(Config):
	# define the name of the configuration
	NAME = "construction_cfg"
	# number of classes (background + construction)
	NUM_CLASSES = 1 + 1
	# number of training steps per epoch
	STEPS_PER_EPOCH = 131


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = 'construction_cfg'
	# number of classes (background + crane)
	NUM_CLASSES = 2
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1


def train_weights():
	# prepare train set
	train_set = NewClassDataset()
	train_set.load_dataset('construction', is_train=True)
	train_set.prepare()
	print('Train: %d' % len(train_set.image_ids))
	# prepare test/val set
	test_set = NewClassDataset()
	test_set.load_dataset('construction', is_train=False)
	test_set.prepare()
	print('Test: %d' % len(test_set.image_ids))
	# prepare config
	config = NewClassConfig()
	config.display()
	# define the model
	model = MaskRCNN(mode='training', model_dir='./', config=config)
	# load weights (mscoco) and exclude the output layers
	model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
	# train weights (output layers or 'heads')
	model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP


def train_new_class():
	# load the train dataset
	train_set = NewClassDataset()
	train_set.load_dataset('construction', is_train=True)
	train_set.prepare()
	print('Train: %d' % len(train_set.image_ids))

	# load the test dataset
	test_set = NewClassDataset()
	test_set.load_dataset('construction', is_train=False)
	test_set.prepare()
	print('Test: %d' % len(test_set.image_ids))

	# prepare config
	config = NewClassConfig()
	config.display()
	model = MaskRCNN(mode='training', model_dir='./', config=config)
	# load weights (mscoco) and exclude the output layers
	model.load_weights('mask_rcnn_coco.h5', by_name=True,
					   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
	# train weights (output layers or 'heads')
	model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')


def predict_new_class():
	# load the train dataset
	train_set = NewClassDataset()
	train_set.load_dataset('construction', is_train=True)
	train_set.prepare()
	print('Train: %d' % len(train_set.image_ids))

	# load the test dataset
	test_set = NewClassDataset()
	test_set.load_dataset('construction', is_train=False)
	test_set.prepare()
	print('Test: %d' % len(test_set.image_ids))

	# create config
	cfg = PredictionConfig()
	# define the model
	model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
	# load model weights
	model.load_weights('./mask_rcnn_construction_cfg_0005.h5', by_name=True)
	# evaluate the model on training dataset
	train_mAP = evaluate_model(train_set, model, cfg)
	print("Train mAP: %.3f" % train_mAP)
	# evaluate model on test dataset
	test_mAP = evaluate_model(test_set, model, cfg)
	print("Test mAP: %.3f" % test_mAP)


# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_image):
	# load the image and mask
	image = dataset.load_image(n_image)
	mask, _ = dataset.load_mask(n_image)
	# convert pixel values (e.g. center)
	scaled_image = mold_image(image, cfg)
	# convert image into one sample
	sample = expand_dims(scaled_image, 0)
	# make prediction
	yhat = model.detect(sample, verbose=0)[0]
	print(yhat)
	# define subplot
	pyplot.subplot(1, 2, 1)
	# plot raw pixel data
	pyplot.imshow(image)
	pyplot.title('Actual')
	# plot masks
	for j in range(mask.shape[2]):
		pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
	# get the context for drawing boxes
	pyplot.subplot(1, 2, 2)
	# plot raw pixel data
	pyplot.imshow(image)
	pyplot.title('Predicted')
	ax = pyplot.gca()
	# plot each box
	for box in yhat['rois']:
		# get coordinates
		y1, x1, y2, x2 = box
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
	# show the figure
	pyplot.show()


def load_custom_image(image_name):
	"""Load the specified image and return a [H,W,3] Numpy array.
    """
	# Load image
	image = skimage.io.imread(image_name)
	# If grayscale. Convert to RGB for consistency.
	if image.ndim != 3:
		image = skimage.color.gray2rgb(image)
	# If has an alpha channel, remove it for consistency
	if image.shape[-1] == 4:
		image = image[..., :3]
	return image


def make_and_save_predictions(dataset, model, cfg, n_image):
	# load the image and mask
	image = dataset.load_image(n_image)
	# convert pixel values (e.g. center)
	scaled_image = mold_image(image, cfg)
	# convert image into one sample
	sample = expand_dims(scaled_image, 0)
	# make prediction
	results = model.detect(sample, verbose=1)[0]
	# Visualize results
	r = results
	boxed_image_name = "boxed_samples/boxed_test_img" + str(n_image) + ".jpg"
	boxed_image = load_custom_image(boxed_image_name)
	image_name = "test_img" + str(n_image)
	save_image(boxed_image, image_name, r['rois'], r['masks'], r['class_ids']
			   , r['scores'], class_names=dataset.class_names, scores_thresh=0.2, mode=1)


def plot_predictions():
	# load the train dataset
	train_set = NewClassDataset()
	train_set.load_dataset('construction', is_train=True)
	train_set.prepare()
	print('Train: %d' % len(train_set.image_ids))

	# load the test dataset
	test_set = NewClassDataset()
	test_set.load_dataset('construction', is_train=False)
	test_set.prepare()
	print('Test: %d' % len(test_set.image_ids))

	# create config
	cfg = PredictionConfig()

	# define the model
	model = MaskRCNN(mode='inference', model_dir='./', config = cfg)

	# load model weights
	model_path = 'mask_rcnn_construction_cfg_0005.h5'
	model.load_weights(filepath=model_path, by_name=True)

	# plot predictions for the test dataset
	# for i in range(len(test_set.image_ids)):
	# 	plot_actual_vs_predicted(test_set, model, cfg, n_image=i)

	# make predictions and save the predictions as images
	for i in range(len(test_set.image_ids)):
		make_and_save_predictions(dataset=test_set, model=model, cfg=cfg, n_image=i)



if __name__ == '__main__':
	plot_predictions()

