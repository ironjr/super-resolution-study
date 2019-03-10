from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as T

def train(model, optimizer, criterion, scale_factor, loader_train,
          loader_val=None, clip_grad=None, device=torch.device('cuda'),
          dtype=None, num_epochs=1, logger_train=None, logger_val=None,
          iteration_begins=0, print_every=100, verbose=True):
    """Trains given model with given optimizer and data loader.

    Args:
        model (:obj:`torch.nn.Module`): A PyTorch Module for a model to be
            trained.
        optimizer (:obj:`torch.optim.optim`): A PyTorch Optimizer defining the
            training method.
        criterion (function): Loss function.
        scale_factor (float): Scale factor of super-resolution.
        loader_train (:obj:`torch.utils.data.DataLoader`): DataLoader having
            training data.
        loader_val (:obj:`torch.utils.data.DataLoader`): DataLoader having
            validation data.
        clip_grad (float, optional): Determine whether to clip gradient for faster
            learning. Does not do clipping if the value is None. Default is None.
        device (:obj:`torch.device`, optional): Device where training is being
            held. Default is CUDA.
        dtype (:obj:`dtype`): Data type of image tensor component. Default is
            None.
        num_epochs (int, optional): Number of epoches to be train.
        logger_train (:obj:`Logger`, optional): Logs history for tensorboard
            statistics. Related to training set. Default is None.
        logger_val (:obj:`Logger`, optional): Logs history for tensorboard
            statistics. Related to validation set. Default is None.
        iteration_begins (int, optional): Tells the logger from where it counts
            the number of iterations passed. Default is 0.
        print_every (int, optional): Period of print of the statistics. Default
            is 100.
        verbose (bool, optional): Print the statistics in detail. Default is
            True.
    
    Returns: Nothing.
    """
    model = model.to(device=device)
    num_steps = len(loader_train)
    for e in range(num_epochs):
        for i, (hrimg, _) in enumerate(loader_train):
            # Model to training mode
            model.train()

            # Rescale the original HR image to generate LR counterpart
            print(hrimg.shape)
            T_resize = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
            ])
            lrimg = T_resize(hrimg)

            if dtype is not None:
                hrimg = hrimg.to(device=device, dtype=dtype)
                lrimg = lrimg.to(device=device, dtype=dtype)
            else:
                hrimg = hrimg.to(device=device)
                lrimg = lrimg.to(device=device)
            
            # Forward path
            out = model(lrimg)
            loss = criterion(out, hrimg)

            # Backward path
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if clip_grad is not None:
                nn.utils.clip_grad_value_(model.parameters(),
                    clip_grad / optimizer.defaults['lr'])

            optimizer.step()

            # Print the intermediate performance. Test for validation data if
            # it is given.
            if verbose and (i + 1) % print_every == 0:
                # Common statistics.
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(e + 1, num_epochs, i + 1, num_steps, loss.item()),
                    end='')
                
                # Validation dataset is provided.
                val_psnr, val_ssim = None, None
                if loader_val is not None and len(loader_val) is not 0:
                    print(', ', end='')
                    val_psnr, val_ssim = test(model, loader_val, device=device,
                        dtype=dtype)
                else:
                    print('')
                
                # Tensorboard logging training set statistics.
                iterations = e * num_steps + i + iteration_begins + 1
                if logger_train is not None:

                    # 1. Scalar summary
                    info = { 'loss': loss.item() }
                    for tag, value in info.items():
                        logger_train.log_scalar(tag, value, iterations)
                    
                    # 2. Historgram summary
                    # for tag, value in model.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     logger_train.log_histogram(tag,
                    #         value.data.cpu().numpy(), iterations)
                    #     logger_train.log_histogram(tag + '/grad',
                    #         value.grad.data.cpu().numpy(), iterations)
                        
                    # 3. Image summary
                    # info = { 'images': x.view(-1, 32, 32)[:10].cpu().numpy()}
                    # for tag, images in info.items():
                    #     logger_train.log_image(tag, images, iterations)

                # Tensorboard logging validation set statistics.
                if logger_val is not None:

                    # 1. Scalar summary
                    info = {}
                    if val_psnr is not None:
                        info['PSNR'] = val_psnr
                    if val_ssim is not None:
                        info['SSIM'] = val_ssim
                    for tag, value in info.items():
                        logger_val.log_scalar(tag, value, iterations)


def test(model, loader_test, device=torch.device('cuda'), dtype=None):
    """Test on singlet without any modification on data.

    Args: 
        model (:obj:`torch.nn.Module`): A PyTorch Module for a model to be
            trained.
        loader_test (:obj:`torch.utils.data.DataLoader`): DataLoader having
            test data.
        device (:obj:`torch.device`, optional): Device where training is being
            held. Default is CUDA.
        dtype (:obj:`dtype`): Data type of image tensor component. Default is
            None.

    Returns:
        Accuracy from the test result.
    """
    # Compute PSNR and SSIM
    avg_psnr = 0
    with torch.no_grad():
        for batch in loader_test:
            if dtype is not None:
                lr = batch[0].to(device=device, dtype=dtype)
                hr = batch[1].to(device=device, dtype=dtype)
            else:
                lr = batch[0].to(device=device)
                hr = batch[1].to(device=device)

            prediction = model(lr)
            mse = F.mse_loss(prediction, hr)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    avg_psnr /= len(loader_test)

    # TODO Compute SSIM
    avg_ssim = None

    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    return avg_psnr, avg_ssim
