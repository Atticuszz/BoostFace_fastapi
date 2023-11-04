def generate_normalized_embeddings(n, dim=512):
    embeddings = np.random.randn(n, dim)  # 生成正态分布的随机向量
    embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]  # 规范化向量
    return embeddings


import numpy as np


def generate_light_image(pixels=None, size=None):
    """
    Generate a LightImage object with either the specified number of pixels or the specified size.
    Include face data with five keypoints as NumPy arrays based on the image size
    or a random rectangle if only pixels are specified.
    """
    from src.boostface.app.common import LightImage
    det_score = 0.8  # Fixed detection score

    if size:
        # If size is specified, create an image with the given size
        height, width = size
        # The entire image is considered a face
        bbox = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.int32)
        # Approximate positions for the five keypoints
        kps = np.array([
            [width // 2, height // 4],  # Nose
            [width // 3, height // 3],  # Left eye
            [2 * width // 3, height // 3],  # Right eye
            [width // 3, 2 * height // 3],  # Left mouth corner
            [2 * width // 3, 2 * height // 3]  # Right mouth corner
        ], dtype=np.int32)
    else:
        # If size is not specified, calculate the size based on the number of pixels
        if pixels is None:
            pixels = 5000000  # Default to 5 million pixels if neither size nor pixels are provided
        side_length = int(np.sqrt(pixels / 3))
        height, width = side_length, side_length
        # Create a random face rectangle in the image
        x1, y1 = np.random.randint(0, width // 2, size=2)
        x2, y2 = np.random.randint(width // 2, width, size=2)
        bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        # Approximate positions for the five keypoints within the bbox
        kps = np.array([
            [(x1 + x2) // 2, (y1 + 3 * y2) // 4],  # Nose
            [x1 + (x2 - x1) // 3, y1 + (y2 - y1) // 3],  # Left eye
            [x1 + 2 * (x2 - x1) // 3, y1 + (y2 - y1) // 3],  # Right eye
            [x1 + (x2 - x1) // 3, y1 + 2 * (y2 - y1) // 3],  # Left mouth corner
            [x1 + 2 * (x2 - x1) // 3, y1 + 2 * (y2 - y1) // 3]  # Right mouth corner
        ], dtype=np.int32)

    image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    face = [bbox, kps, det_score, None, None]
    faces = [face]

    return LightImage(nd_arr=image_array, faces=faces)


if __name__ == "__main__":
    img = generate_light_image(size=(640, 640))
    print(img)
