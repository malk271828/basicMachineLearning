from random import seed
from random import random, randrange
# seed random number generator
seed(1)

# Image Processing
from PIL import Image, ImageDraw, ImageChops

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler

def test_image():
    WIDTH = 500
    HEIGHT = 300
    src = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))

    for i in range(30):
        im = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        alpha = int(100)
        draw.rectangle(xy=[(randrange(WIDTH), randrange(HEIGHT)), (100, 100)], fill=(alpha, alpha, alpha))

        result = ImageChops.add(src, im)

    result.save('image.png', quality=95)
