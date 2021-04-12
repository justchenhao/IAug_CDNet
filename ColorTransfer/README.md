# Color Transfer

We resort to a simple yet effective nonlearning approach to match the color distribution of the two image sets (GAN-generated images and original images in the change detection dataset).

Color Transfer between Images in a correlated colour space ( RGB ).

![color_transfer](./images/color_transfer.png)

## requirements

- Matlab

## usage

We provide two demos to show the color transfer. 

When you do not have the object mask. You can edit the file `ColorTransferDemo.m`, modify the file path of the `Im_target` and `Im_source`. After you run this file, the transfered image is saved as `result_Image.jpg`.

When you do have both the building image and the object mask. You can edit the file `ColorTransferDemo_with_mask.m`, modify the file path of the `Im_target`, `Im_source`, `m_target` and `m_source`. After you run this file, the transfered image is saved as `result_Image.jpg`.

## Acknowledge 

This code borrows heavily from https://github.com/AissamDjahnine/ColorTransfer. 