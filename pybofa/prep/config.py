
# section: image data
class data:
    gh_admin = r'C:/Users/jagmeet/bofa_data/final_control_df.csv'
    

    # section: image meta data
    mild_count = 1792
    moderate_count = 724
    no_count = 2560
    data_folders = ["mild", "moderate", "no"]


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





