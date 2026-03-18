
# section: image data
class data:
    merged_df = r'C:/Users/jagmeet/bofa_data/merged_df.csv'


# section: image processor
class processor:    
    batch_size = 32
    img_size = (224,224)
    validation_split = 0.2
    clipLimit = 1.2
    tileGridSize = (8,8)

#section: model
class model:
    network_layers = [32, 64, 128, 256]
    activation = 'softmax'
    negative_slope = 0.01
    learning_rate = 1e-5
    clipnorm = 1.0

    #early stopping check system
    patience = 20

    #model fit
    epoch=50
    epochs_2 = 10
    dim = 9





