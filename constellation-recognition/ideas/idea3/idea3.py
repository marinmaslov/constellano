from PIL import Image, ImageDraw, ImageFont
from numpy import ndarray
import numpy as np
from scipy.spatial.distance import euclidean
 
BLOCK_SIZE = 120
 
def mark_blocks(image):
    vertical_blocks, horizontal_blocks = image.size[1] // BLOCK_SIZE, image.size[0] // BLOCK_SIZE
    current = 1
 
    for i in range(vertical_blocks):
        for j in range(horizontal_blocks):
            draw = ImageDraw.Draw(image)
            draw.rectangle(((j * BLOCK_SIZE, i * BLOCK_SIZE),
                            (j * BLOCK_SIZE + BLOCK_SIZE, i * BLOCK_SIZE + BLOCK_SIZE)))
            draw.text((j * BLOCK_SIZE, i * BLOCK_SIZE), str(current))
            current += 1
 
    image.show()
 
def get_image_blocks(image, display_image=False):
    np_image = np.asarray(image)
    cols, rows = np_image.shape[0], np_image.shape[1]
 
    if display_image:
        mark_blocks(image)
 
    for r in range(0, rows - BLOCK_SIZE + 1, BLOCK_SIZE):
        for c in range(0, cols - BLOCK_SIZE + 1, BLOCK_SIZE):
            yield np_image[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE]
 
def calculate_lbp_histogram(block: ndarray):
    histogram = [0] * 256
    cols, rows = block.shape[0], block.shape[1]
 
    def calculate_single_lbp(i: int, j: int):
        bin_vals = [0] * 8
        i_size, j_size = block.shape[0], block.shape[1]
        try:
            bin_vals[0] = int(block[i, j] < (
                block[i, j + 1] if j + 1 < j_size else 0))
            bin_vals[1] = int(block[i, j] < (
                block[i + 1, j + 1] if j + 1 < j_size and i + 1 < i_size else 0))
            bin_vals[2] = int(block[i, j] < (
                block[i + 1, j] if i + 1 < i_size else 0))
            bin_vals[3] = int(block[i, j] < (
                block[i + 1, j - 1] if j - 1 > 0 and i + 1 < i_size else 0))
            bin_vals[4] = int(block[i, j] < (
                block[i, j - 1] if j - 1 > 0 else 0))
            bin_vals[5] = int(block[i, j] < (
                block[i - 1, j - 1] if j - 1 > 0 and i - 1 > 0 else 0))
            bin_vals[6] = int(block[i, j] < (
                block[i - 1, j] if i - 1 > 0 else 0))
            bin_vals[7] = int(block[i, j] < (
                block[i - 1, j + 1] if j + 1 < j_size and i - 1 > 0 else 0))
        finally:
            return int(''.join(map(lambda x: str(x), bin_vals)), base=2)
    for i in range(rows):
        for j in range(cols):
            px_lbp = calculate_single_lbp(i, j)
            histogram[px_lbp] += 1
    return histogram
 
def main():
    slika = Image.open("../../img/ursa_major_1.jpg")
    slika_grayscale = slika.convert('L')
    image_blocks = get_image_blocks(slika_grayscale, display_image=True)
    block_histograms = []
    for block in image_blocks:
        block_histograms.append(calculate_lbp_histogram(block))
 
    nearest_neighbours = {}
    for i in range(len(block_histograms)):
        for j in range(len(block_histograms)):
            if i == j:
                continue
            if i not in nearest_neighbours:
                nearest_neighbours[i] = (
                    euclidean(block_histograms[i], block_histograms[j]), j)
            else:
                candidate = euclidean(block_histograms[i], block_histograms[j])
                if candidate < nearest_neighbours[i][0]:
                    nearest_neighbours[i] = (candidate, j)
    for k in {k: v for k, v in sorted(nearest_neighbours.items(), key=lambda kv: kv[1])}:
        print(
            f'Nearest neighbour of {k + 1} is {nearest_neighbours[k][1] + 1} with distance of {nearest_neighbours[k][0]}')
 
if __name__ == '__main__':
    main()
