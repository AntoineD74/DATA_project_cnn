from tensorflow.keras import layers, models
import config

def resnet_block(input, filters, kernel_size, reduce=False):
    """
        Implementation of a Resnet block which outptu can be added to another layer deeper in the network
    """
    result=layers.Conv2D(filters, kernel_size, strides=1, padding='SAME')(input)
    result=layers.BatchNormalization()(result)
    result=layers.LeakyReLU(alpha=0.1)(result)

    if reduce is True:
        result=layers.Conv2D(filters, kernel_size, strides=2, padding='SAME')(result)
    else:
        result=layers.Conv2D(filters, kernel_size, strides=1, padding='SAME')(result)
        
    if input.shape[-1]==filters:
        if reduce is True:
            shortcut=layers.Conv2D(filters, 1, strides=2, padding='SAME')(input)
        else:
            shortcut=input
    else:
        if reduce is True:
            shortcut=layers.Conv2D(filters, 1, strides=2, padding='SAME')(input)
        else:
            shortcut=layers.Conv2D(filters, 1, strides=1, padding='SAME')(input)
    
    result=layers.add([result, shortcut])
    result=layers.LeakyReLU(alpha=0.1)(result)
    result=layers.BatchNormalization()(result)
    return result


def model(nbr_classes, nbr_boxes, cellule_y, cellule_x):
    """
        Implementation of a Resnet model
    """
    inputs=layers.Input(shape=(config.largeur, config.hauteur, 3), dtype='float32')

    result=resnet_block(inputs, 16, 3, False)
    result=resnet_block(result, 16, 3, True)

    result=resnet_block(result, 32, 3, False)
    result=resnet_block(result, 32, 3, True)

    result=resnet_block(result, 64, 3, False)
    result=resnet_block(result, 64, 3, False)
    result=resnet_block(result, 64, 3, True)

    result=resnet_block(result, 128, 3, False)
    result=resnet_block(result, 128, 3, False)
    result=resnet_block(result, 128, 3, True)

    result=layers.Conv2D(config.nbr_boxes*(5+config.nbr_classes), 1, padding='SAME')(result)
    output=layers.Reshape((config.cellule_y, config.cellule_x, config.nbr_boxes, 5+config.nbr_classes))(result)

    model=models.Model(inputs=inputs, outputs=output)
    return model

'''
model_cnn = model(10, 100, config.cellule_y, config.cellule_x)
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
'''