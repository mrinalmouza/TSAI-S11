'''
Utility script for modular code
'''
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def misclassified_images(model, dataset, images_count, device, criterion):

    model.eval()
    
    with torch.no_grad():
        
        for data, target in dataset:

            data, target = data.to(device), target.to(device)
           
            output = model(data)
            
            #Get the predictions
            _, pred = torch.max(output.data, 1)
            
            #Below code gives a MASK of true and false for images that have been correctly classified as True otherwise False
            correct_preds = pred.eq(target.view_as(pred))
            
            #Here we are plucking out all the missclassfiied images. The mask select all the images that are false(meaning missclassified)
            incorrect_preds = data[~correct_preds]
            
            #only taking what we need
            misclassified_images = incorrect_preds[0:images_count, :, :, :]
            
            #Getting the labels of predicted images
            predicted_labels = pred[~correct_preds][0:images_count]
            #Getting actual labels
            actual_labels = target.view_as(pred)[~correct_preds][0:images_count]

    return(misclassified_images,predicted_labels,actual_labels)


#inverse normalization
# Inverse normalization is just rearranging of means and std to get the orgininal image back
#Normalized Image = (Un-Normalized image - Mean)/Std
#Un-normalized image = Normalized image * std + Mean -->> std(NI + Mean/std) --> (NI - (-Mean/std))/ (1/std)
# We need to convert into the above format so that it can fit the formula that A.Normalize uses.
def inverse_normalize(images):

    normalize_transform = transforms.Normalize(mean = (-0.49/0.24, -0.48/0.24, -0.44/0.26),std = (1/0.24, 1/0.24, 1/0.26))
    Unormalized_images = normalize_transform (images)
    Unormalized_images = torch.clamp(Unormalized_images, 0, 1)
    return (Unormalized_images)


def show_misclassfied_images(images, inttoclasses,a_labels,p_labels):
    Unormalized_images = inverse_normalize (images)
    # Print 12 images to see the sample data
    plt.figure(figsize=(10,10))
    for i in range(20):
        plt.subplot(4,5,i+1)
        im = Unormalized_images[i].cpu()
        im = np.transpose(im, axes=[1, 2, 0])
        plt.imshow(im)
        #plt.title(inttoclasses[p_labels[i].item()])
        plt.title('actual:' + {inttoclasses[a_labels[i].item()]} + '\n' +
                    'predicted:' + {inttoclasses[p_labels[i].item()]})
        plt.axis('off')
    
def show_grad_cam_images(model, target_layers, images, inttoclasses, a_labels, p_labels):
    fig = plt.figure(figsize=(20, 20))
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=images, targets=None)

    grayscale_cam = grayscale_cam[0, :]

    image_batch = inverse_normalize(images.to('cpu'))

    for i in range(0, len(images)-1):
        #pdb.set_trace()
        plt.subplot(5, 4, i + 1)
        rgb_img = np.transpose(image_batch[i], (1, 2, 0))

        rgb_img = rgb_img.numpy()

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True,image_weight=0.70)

        plt.imshow(visualization)
        plt.title(r"Correct: " + inttoclasses[a_labels[i].item()] + '\n' + 'Output: ' + inttoclasses[p_labels[i].item()])
        plt.xticks([])
        plt.yticks([])