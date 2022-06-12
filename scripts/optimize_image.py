import torch.nn
from torch import nn, optim, Tensor
from torchvision import models, transforms
import os
from modelling.local_model_store import LocalModelStore
import argparse
from PIL import Image
from tqdm import tqdm
import const


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_filename", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--architecture", type=str, default='vgg16')
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--model_weights_path", type=str)
    parser.add_argument("--config_dir", type=str, default=None)
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--freq", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=8749)
    parser.add_argument("--optimized_class", type=int, default=0)
    parser.add_argument("--output_path", type=int)

    args = parser.parse_args()
    return args

img_to_tensor = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
tensor_to_image = transforms.Compose([
    transforms.Normalize([-1, -1, -1], [2, 2, 2]),
    transforms.ToPILImage()
])

def load_image(im_path: str) -> Tensor:
    im1 = Image.open(im_path)

    if im1.mode != 'RGB':
        im1 = im1.convert('RGB')

    im1t = img_to_tensor(im1)
    im1t = im1t.unsqueeze(0)

    if torch.cuda.is_available() and const.DEBUG is False:
        im1t = im1t.cuda()

    return im1t


def optimize_image(model: nn.Module, cls: int, image: Tensor, iters: int, output_folder: str, freq: int = 100) -> None:
    fc8 = model.classifier[-1].__dict__['_parameters']['weight']
    prototype_vec = fc8[cls]

    model.requires_grad_(False)
    model.train(False)
    image.requires_grad = True
    del model.classifier[-1]
    # del model.classifier[-1]
    # del model.classifier[-1]
    prototype_vec = prototype_vec.detach().clone()
    prototype_vec = nn.ReLU()(prototype_vec)
    cos_sim = torch.nn.CosineSimilarity(0)

    cos_dist = lambda v1, v2: 1 - cos_sim(v1, v2)
    optimizer = torch.optim.SGD([image], lr=0.1, momentum=0.9)
    lr_reducer = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1000)
    pbar = tqdm(range(iters), desc='iters')
    for i in pbar:
        optimizer.zero_grad()
        rep_vec = model(image)
        rep_vec = rep_vec.squeeze()
        loss = cos_dist(rep_vec, prototype_vec)
        pbar.set_description(f'cos_dist={float(loss)}')
        loss.backward()
        optimizer.step()
        lr_reducer.step(loss)
        if i % freq == 0:
            pil_image = tensor_to_image(image.squeeze(0))
            pil_image.save(os.path.join(output_folder, f'{i}.png'))


if __name__ == '__main__':
    args = get_args()
    cls = args.optimized_class
    model = models.vgg16(num_classes=args.num_classes)
    get_args()

    im_name = args.image_filename
    im_path = os.path.join(args.image_dir, im_name)
    image = load_image(im_path)

    output_dir = os.path.join(args.output_path, str(cls), im_name)
    os.makedirs(output_dir, exist_ok=True)
    model.features = nn.DataParallel(model.features)
    if const.DEBUG is False:
        model.cuda()
    LocalModelStore(args.architecture, args.experiment_name, None).load_model_and_optimizer_loc(model=model, model_location=args.model_weights_path)
    optimize_image(model, cls, image, args.iters, output_dir, freq=args.freq)
