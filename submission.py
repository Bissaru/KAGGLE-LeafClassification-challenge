import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Assuming the model is already loaded in the variable 'model'
# Set the path to the directory containing the test images
test_images_dir = 'Data/test_images'

# List of class names
class_names = [
    "Acer_Capillipes", "Acer_Circinatum", "Acer_Mono", "Acer_Opalus", "Acer_Palmatum", "Acer_Pictum",
    "Acer_Platanoids", "Acer_Rubrum", "Acer_Rufinerve", "Acer_Saccharinum", "Alnus_Cordata", "Alnus_Maximowiczii",
    "Alnus_Rubra", "Alnus_Sieboldiana", "Alnus_Viridis", "Arundinaria_Simonii", "Betula_Austrosinensis",
    "Betula_Pendula", "Callicarpa_Bodinieri", "Castanea_Sativa", "Celtis_Koraiensis", "Cercis_Siliquastrum",
    "Cornus_Chinensis", "Cornus_Controversa", "Cornus_Macrophylla", "Cotinus_Coggygria", "Crataegus_Monogyna",
    "Cytisus_Battandieri", "Eucalyptus_Glaucescens", "Eucalyptus_Neglecta", "Eucalyptus_Urnigera", "Fagus_Sylvatica",
    "Ginkgo_Biloba", "Ilex_Aquifolium", "Ilex_Cornuta", "Liquidambar_Styraciflua", "Liriodendron_Tulipifera",
    "Lithocarpus_Cleistocarpus", "Lithocarpus_Edulis", "Magnolia_Heptapeta", "Magnolia_Salicifolia", "Morus_Nigra",
    "Olea_Europaea", "Phildelphus", "Populus_Adenopoda", "Populus_Grandidentata", "Populus_Nigra", "Prunus_Avium",
    "Prunus_X_Shmittii", "Pterocarya_Stenoptera", "Quercus_Afares", "Quercus_Agrifolia", "Quercus_Alnifolia",
    "Quercus_Brantii", "Quercus_Canariensis", "Quercus_Castaneifolia", "Quercus_Cerris", "Quercus_Chrysolepis",
    "Quercus_Coccifera", "Quercus_Coccinea", "Quercus_Crassifolia", "Quercus_Crassipes", "Quercus_Dolicholepis",
    "Quercus_Ellipsoidalis", "Quercus_Greggii", "Quercus_Hartwissiana", "Quercus_Ilex", "Quercus_Imbricaria",
    "Quercus_Infectoria_sub", "Quercus_Kewensis", "Quercus_Nigra", "Quercus_Palustris", "Quercus_Phellos",
    "Quercus_Phillyraeoides", "Quercus_Pontica", "Quercus_Pubescens", "Quercus_Pyrenaica", "Quercus_Rhysophylla",
    "Quercus_Rubra", "Quercus_Semecarpifolia", "Quercus_Shumardii", "Quercus_Suber", "Quercus_Texana", "Quercus_Trojana",
    "Quercus_Variabilis", "Quercus_Vulcanica", "Quercus_x_Hispanica", "Quercus_x_Turneri", "Rhododendron_x_Russellianum",
    "Salix_Fragilis", "Salix_Intergra", "Sorbus_Aria", "Tilia_Oliveri", "Tilia_Platyphyllos", "Tilia_Tomentosa",
    "Ulmus_Bergmanniana", "Viburnum_Tinus", "Viburnum_x_Rhytidophylloides", "Zelkova_Serrata"
]

# Define the size to which the images will be resized
img_size = (256, 256)  # Adjust this based on your model's input size


model = load_model("callback_logs")
# Prepare a list to hold the results
results = []

# Process each image in the test directory
for img_name in os.listdir(test_images_dir):
    # Extract the numeric part of the filename to use as id
    img_id = int(os.path.splitext(img_name)[0])
    
    # Load the image in grayscale mode
    img_path = os.path.join(test_images_dir, img_name)
    img = load_img(img_path, target_size=img_size, color_mode='grayscale')
    img_array = img_to_array(img) / 255.0  # Normalize the image to [0, 1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension if necessary

    # Predict the class probabilities
    predictions = model.predict(img_array)[0]

    # Prepare the result row
    result_row = [img_id] + predictions.tolist()
    results.append(result_row)

# Create a DataFrame with the results
columns = ['id'] + class_names
df = pd.DataFrame(results, columns=columns)

# Save the DataFrame to a CSV file
output_csv_path = 'Data/output_predictions.csv'
df.to_csv(output_csv_path, index=False)