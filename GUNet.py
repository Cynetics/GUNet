""" 
This model was tested on Rotation MNIST
Essentially, it's a UNet with Group convolutions from the input
to the output. The model is as of now hardcoded for 2 down and 2 up convolutions.
That is, the size of the image gets encoded twice with a stride of two.
Suppose a 32x32 input image, then the model will encode and decode the model as:

32x32 -> 16x16 -> 8x8 -> 16x16 -> 32x32

TODO:

make the GUNet general so we can use it for other input sizes.

Credit: https://github.com/QUVA-Lab/e2cnn

"""

import torch

from e2cnn import gspaces
from e2cnn import nn

class GUNet(torch.nn.Module):
    # MNIST GUNet
    def __init__(self, n_classes=10, feature_fields=6, classes=4):

        super(GUNet, self).__init__()

        ################# ENCODER ############################################

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        self.feature_fields = feature_fields
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 6 feature fields, each transforming under the regular representation of C8
        out_type_1 = nn.FieldType(self.r2_act, feature_fields*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 32, margin=1),
            nn.R2Conv(in_type, out_type_1, kernel_size=5, padding=2, stride=1, bias=False),
            nn.InnerBatchNorm(out_type_1),
            nn.ReLU(out_type_1, inplace=True)
        )

        # convolution 2
        # the old output type is the input type to the next layer
        # the output type of the second convolution layer are 12 regular feature fields of C8
        out_type_2 = nn.FieldType(self.r2_act, feature_fields*2*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(out_type_1, out_type_2, kernel_size=5, padding=2, stride=1, bias=False),
            nn.InnerBatchNorm(out_type_2),
            nn.ReLU(out_type_2, inplace=True)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type_2, sigma=0.66, stride=2)
        )

        # convolution 3
        # the old output type is the input type to the next layer
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type_3 = nn.FieldType(self.r2_act, feature_fields*4*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(out_type_2, out_type_3, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type_3),
            nn.ReLU(out_type_3, inplace=True)
        )

        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type_3, sigma=0.66, stride=2)
        )

        ############################## DECODER ######################################################

        out_type_up_1 = nn.FieldType(self.r2_act, feature_fields*4*[self.r2_act.regular_repr])

        self.center = nn.SequentialModule(
                        nn.R2Upsampling(out_type_3, scale_factor=2, mode='bilinear', align_corners=True),
                        nn.R2Conv(out_type_3, out_type_up_1, kernel_size=5, padding=2, bias=False),
                        nn.InnerBatchNorm(out_type_up_1),
                        nn.ReLU(out_type_up_1, inplace=True))

        out_type_up_2 = nn.FieldType(self.r2_act, feature_fields*2*[self.r2_act.regular_repr])

        self.up4 = nn.SequentialModule(
                        nn.R2Upsampling(out_type_up_1+out_type_3, scale_factor=2, mode='bilinear', align_corners=True),
                        nn.R2Conv(out_type_up_1+out_type_3, out_type_up_2, kernel_size=5, padding=2, bias=False),
                        nn.InnerBatchNorm(out_type_up_2),
                        nn.ReLU(out_type_up_2, inplace=True))

        out_type_up_3 = nn.FieldType(self.r2_act, feature_fields*2*[self.r2_act.regular_repr])

        self.up3 = nn.SequentialModule(
                        nn.R2Upsampling(out_type_up_2+out_type_2, scale_factor=1, mode='bilinear', align_corners=True),
                        nn.R2Conv(out_type_up_2+out_type_2, out_type_up_3, kernel_size=5, padding=1, bias=False),
                        nn.InnerBatchNorm(out_type_up_3),
                        nn.ReLU(out_type_up_3, inplace=True))

        ########################## OUTPUT ########################

        out_type_up_3 = nn.FieldType(self.r2_act, feature_fields*2*[self.r2_act.regular_repr])
        self.out_block = nn.SequentialModule(
            nn.R2Conv(out_type_up_2+out_type_2, out_type_up_3, kernel_size=5, padding=2, stride=1, bias=False),
            nn.InnerBatchNorm(out_type_up_3),
            nn.ReLU(out_type_up_3, inplace=True)
        )

        out_type_seg = nn.FieldType(self.r2_act, classes*[self.r2_act.regular_repr])
        self.segmentation_head = nn.SequentialModule(
            nn.R2Conv(out_type_up_3, out_type_seg, kernel_size=1, padding=0, stride=1, bias=False)
        )

        self.gpool = nn.GroupPooling(out_type_seg)
        # number of output channels
        c = self.gpool.out_type.size

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types

        # ENCODE
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.pool1(x2)
        x4 = self.block3(x3)
        x5 = self.pool2(x4)

        # CENTER of the GUNet
        xc = self.center(x5)

        # DECODE
        concat_type_up4 = nn.FieldType(self.r2_act, self.feature_fields*8*[self.r2_act.regular_repr])
        xup1 = self.up4(nn.GeometricTensor(torch.cat([xc.tensor,x4.tensor],dim=1), concat_type_up4))

        concat_type_up2 = nn.FieldType(self.r2_act, self.feature_fields*4*[self.r2_act.regular_repr])
        x = self.out_block(nn.GeometricTensor(torch.cat([xup1.tensor,x2.tensor],dim=1), concat_type_up2))
        x = self.segmentation_head(x)

        # pool over the group
        x = self.gpool(x)
        x = x.tensor
        return x
