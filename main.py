import argparse
import os

import cv2.cv2
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import numpy as np
import torch.nn.functional as F

import eval_sets
import util

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='Device for evaluating networks.')
parser.add_argument('--model_name', type=str, default='vgg16', required=False, help='Target model to use.')
parser.add_argument('--smodel_name', type=str, default='inception_v3, resnet50',
                    help='One or more surrogate models to use (enter all names, separated by spaces).')
parser.add_argument('--targeted', action='store_true', help='If true, perform targeted attack; else, untargeted.')
parser.add_argument('--min_attack_samples', type=int, default=2, help="Number of 'outer' SimBA iterations. Note that each "
                                                             "iteration may consume 1 or 2 queries.")
parser.add_argument('--num_sample', default=10, type=int, help='Number of sample images to attack.')

parser.add_argument('--output', required=False, default='out', help='Name of the output file.')
parser.add_argument('--norm_bound', type=float, default=float('inf'),
                    help='Radius of l2 norm ball onto which solution will be maintained through PGD-type optimisation. '
                         'If not supplied, is effectively infinite (norm is unconstrained).')
parser.add_argument('--net_specific_resampling', action='store_true',
                    help='If specified, resizes input images to match expectations of target net (as always), but adds '
                         'a linear interpolation step to each surrogate network to match its expected resolution. '
                         'Gradients are thus effectively computed in the native surrogate resolutions and returned to '
                         'the target net''s own resolution via the reverse interpolation.')

args = parser.parse_args()

epsilons = [0, .05, .1, .15, .2, .25, .3]

output_folder = '../datasets/pertubated_images'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
mean = util.imagenet_mean
std = util.imagenet_std

pretrained_model = getattr(models, args.model_name)(pretrained=True)
model = torch.nn.Sequential(
    util.Normalise(mean, std),
    pretrained_model
)
model.to(device).eval()

smodel_name = args.smodel_name.split(',')
smodel_name = [s.strip() for s in smodel_name]

surrogate_model_list = []
for s in range(len(smodel_name)):
    pretrained_model = getattr(models, smodel_name[s])(pretrained=True)
    if args.net_specific_resampling:
        # Note that this is, by necessity, case-by-case. If using any nets other than inception_v3 that use input
        # resolutions other than 224x224, they must be added here.
        image_width = 299 if smodel_name[s] == 'inception_v3' else 224
        pretrained_model = torch.nn.Sequential(
            util.Interpolate(torch.Size([image_width, image_width]), 'bilinear'),
            util.Normalise(mean, std),
            pretrained_model
        )
    else:
        pretrained_model = torch.nn.Sequential(
            util.Normalise(mean, std),
            pretrained_model
        )
    surrogate_model_list.append(pretrained_model.to(device).eval())

loss_func = torch.nn.functional.cross_entropy if args.targeted else util.margin_loss

data_transform, image_width = util.generate_data_transform(
    "imagenet_inception_299" if args.model_name == "inception_v3" else "imagenet_common_224"
)

# Set your ImageNet folder path here. Consult the documentation for torchvision.datasets.ImageNet to understand what
# files must be placed where initially. Only the val set is required here.
# imagenet_path = '/your/imagenet/dataset/path'
imagenet_path = '../datasets/imagenet'
dataset = datasets.ImageNet(imagenet_path, split='val', transform=data_transform)

# if args.data_index_set == 'imagenet_val_random':
# input_index_list = torch.randperm(len(dataset))[:args.num_sample]
# else:
#     input_index_list = getattr(eval_sets, args.data_index_set)[:args.num_sample]

if args.targeted:
    target_class_list = []

for i, s in enumerate(range(len(dataset))):
    (image, label) = dataset[s]
    image.unsqueeze_(0)
    label = torch.LongTensor([label])

    image = image.to(device)
    label = label.to(device)
    label_attacked = label.clone()

    attacked_samples = []

    if args.targeted:
        label_attacked[0] = util.any_imagenet_id_but(label.item())

    logits = model(image).data
    to_attack = (torch.argmax(logits, dim=1) != label_attacked) if args.targeted else (
            torch.argmax(logits, dim=1) == label_attacked)
    if to_attack:
        X_best = image.clone()
        if args.targeted:
            loss_best = -loss_func(logits, label_attacked)
            class_org = label[0].item()
            class_tgt = label_attacked[0].item()
        else:
            loss_best, class_org, class_tgt = loss_func(logits.data, label_attacked)
        nQuery = 1  # query for the original image

        for ind, surrogate_model in enumerate(surrogate_model_list):
            for epsilon in epsilons:
                X_grad = X_best.detach().clone().requires_grad_()
                output = surrogate_model(X_grad)
                loss = F.nll_loss(output, label)
                model.zero_grad()
                loss.backward()

                data_grad = X_grad.grad.data
                attacked_image = fgsm_attack(X_grad, epsilon, data_grad)

                logits = model(attacked_image).data
                class_attacked = torch.argmax(logits, dim=1)

                if label != class_attacked:
                    attacked_samples.append(attacked_image)  # TODO: add documentation on which model it was attacked on?
                    break


        if len(attacked_samples) >= args.min_attack_samples:
            image_data_path = os.path.join(output_folder, 'sample_' + str(i).zfill(6))
            if not os.path.exists(image_data_path):
                os.mkdir(image_data_path)

            for idx, pertubated_image in enumerate(attacked_samples):
                image_path = os.path.join(image_data_path, 'pertubation_' + str(idx).zfill(2) + '.jpg')
                cv2.imwrite(image_path, np.array(pertubated_image.squeeze(dim=0).detach().numpy() * 255, dtype=np.uint8).transpose(1, 2, 0))
        else:
            continue