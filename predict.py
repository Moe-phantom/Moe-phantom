import argparse
import torch
import numpy as np
import json
from torch import nn
from torchvision import transforms
from PIL import Image

def load_model(model_checkpoint):
    model_info = torch.load(model_checkpoint)
    model = model_info['model']
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    return model

def process_image(image):
    im = Image.open(image)
    width, height = im.size
    picture_coords = [width, height]
    max_span = max(picture_coords)
    max_element = picture_coords.index(max_span)
    if max_element == 0:
        min_element = 1
    else:
        min_element = 0
    aspect_ratio = picture_coords[max_element] / picture_coords[min_element]
    new_picture_coords = [0, 0]
    new_picture_coords[min_element] = 256
    new_picture_coords[max_element] = int(256 * aspect_ratio)
    im = im.resize(new_picture_coords)
    width, height = new_picture_coords
    left = (width - 244) / 2
    top = (height - 244) / 2
    right = (width + 244) / 2
    bottom = (height + 244) / 2
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
    np_image = np_image.astype('float64')
    np_image = np_image / [255, 255, 255]
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def classify_image(image_path, model, topk=5, gpu=False):
    topk = int(topk)
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        if gpu:
            image = image.cuda()
            model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].tolist()
        results = zip(probs, classes)
        return results

def read_categories(category_names):
    if category_names is not None:
        with open(category_names, 'r') as f:
            jfile = json.load(f)
        return jfile
    return None

def display_prediction(results, category_names):
    cat_file = read_categories(category_names)
    i = 0
    for p, c in results:
        i = i + 1
        p = str(round(p, 4) * 100.) + '%'
        if cat_file:
            c = cat_file.get(str(c), 'None')
        else:
            c = ' class {}'.format(str(c))
        print("{}.{} ({})".format(i, c, p))
    return None

def parse():
    parser = argparse.ArgumentParser(description='Use a neural network to classify an image!')
    parser.add_argument('image_input', help='Image file to classify (required)')
    parser.add_argument('model_checkpoint', help='Model used for classification (required)')
    parser.add_argument('--top_k', help='How many prediction categories to show [default 5].')
    parser.add_argument('--category_names', help='File for category names')
    parser.add_argument('--gpu', action='store_true', help='GPU option')
    args = parser.parse_args()
    return args

def main():
    args = parse()
    if args.gpu and not torch.cuda.is_available():
        raise Exception("--gpu option enabled...but no GPU detected")
    if args.top_k is None:
        top_k = 5
    else:
        top_k = args.top_k
    model = load_model(args.model_checkpoint)
    prediction = classify_image(args.image_input, model, top_k, args.gpu)
    display_prediction(prediction, args.category_names)
    return prediction

if __name__ == "__main__":
    main()
