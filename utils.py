from PIL import Image

import torch
from torchvision import transforms


def load_image(path, max_size=480, shape=None):
    """
    load an image, resize it, convert it to a tensor,
    return the normalized tensor with appropriate dimensions.
    """
    image = Image.open(path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    # transforms to apply to the image
    im_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    # discard the transparent (or) alpha channel (that's the :3)
    # and add the batch dimension
    image = im_transform(image)[:3, :, :].unsqueeze(0)

    return image


def im_convert(tensor):
    """ display a tensor as an image. """

    # bring it to cpu for numpy operations
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()  # convert to np img & remove the batch dim
    image = image.transpose(1, 2, 0)  # C, H, W --> H, W, C
    # image *= np.array((0.229, 0.224, 0.225)) + \
    #     np.array((0.485, 0.456, 0.406))  # un-normalize the image
    image = image.clip(0, 1)  # clip the values between 0 and 1

    return image


def gram_matrix(tensor):
    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()

    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram
