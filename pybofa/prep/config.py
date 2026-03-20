
# section: image data
class data:
    merged_df = r'C:/Users/jagmeet/bofa_data/merged_df.csv'


# section: image processor
class processor:    
    validation_split = 0.2


#section: model
class model:
       #model fit
    epochs=50
    dim = 9

class isolation_forest:
    contam = 0.05
    estimators = 900 # the number of trees and parititoning
    top = 10
    





