# sklearn skimage
from skimage import feature

# Histogram of Oriented Gradients
class HOG:
	def __init__(self, orientations = 9, pixelsPerCell = (9, 9),
		cellsPerBlock = (3, 3), block_norm = 'L2-Hys'):  
		self.orientations = orientations
		self.pixelsPerCell = pixelsPerCell
		self.cellsPerBlock = cellsPerBlock
		self.block_norm = block_norm    # changing from default to L2-Hys, improved a lot

	def describe(self, image):
		# compute HOG for the image
		hist = feature.hog(image, orientations = self.orientations,
			pixels_per_cell = self.pixelsPerCell,
			cells_per_block = self.cellsPerBlock,
			block_norm = self.block_norm) 

		# return the HOG features
		return hist