# 1 Recommender Model

Importing libraries
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np
import ast
from keras.optimizers import AdamW
import random
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

tf.__version__

"""Importing training dataset and pre-processing it"""

# Load Excel files into pandas DataFrames
# creatives_df = pd.read_csv('/content/llb_merged_output.csv')
creatives_df = pd.read_csv('/content/swarm_roas_final.csv')
audience_df = pd.read_csv('/content/swarm_roas_final.csv')
ctr_df = pd.read_csv('/content/swarm_roas_final.csv')


creatives_df['Creative_embeddings'] = creatives_df['Creative_embeddings'].fillna('[]')
# creatives_df['Creatives_embedding'] = creatives_df['Averaged_Embedding'].fillna('[]')  # Replace 'nan' with empty list

#creatives_df['Creatives_embedding'] = creatives_df['Embedding'].fillna('[]')  # Replace 'nan' with empty list
# Extract and process the relevant columns and Pad the embeddings to ensure they have the same length
max_length = max(len(eval(embed)) for embed in creatives_df['Creative_embeddings'])
X_creatives = np.array([eval(embed) + [0] * (max_length - len(eval(embed))) for embed in creatives_df['Creative_embeddings']])# The embeddings are padded with zeros to make them the same length



# Convert the Audience embedding column to a list of lists
def parse_embedding(embedding_str):
    # Remove brackets, split the string, and handle newlines
    embedding_str = embedding_str.strip('[]')  # Remove brackets
    components = embedding_str.replace('\n', '').split() # Split by whitespace to separate numbers
    # Convert the components to floats, handling potential errors
    return [float(comp) for comp in components if comp.strip()] # Ignore empty strings

X_audiences_list = [parse_embedding(embed) for embed in audience_df['Summed Embedding']]

# Find the maximum length of audience embeddings
max_audience_length = max(len(embed) for embed in X_audiences_list)

# Pad audience embeddings to have the same length
X_audiences_padded = [list(embed) + [0] * (max_audience_length - len(embed)) for embed in X_audiences_list]
# Now convert the list of lists to a numpy array
X_audiences = np.array(X_audiences_padded)  # Use the padded list


# x_ctr=ctr_df['label']
# Extract the CTR values as a numpy array
y_ctr = ctr_df['label'].to_numpy()
# X_creatives_combined = np.concatenate([X_creatives, X_objectives, X_goal], axis=1)

# X_combined now contains the concatenated embeddings of creatives, objectives, and goals
# print(X_creatives_combined.shape)

np.save('creatives_embeddings.npy', X_creatives)
np.save('audience_embeddings.npy', X_audiences)
np.save('ctr_values.npy', y_ctr)
print(X_creatives.shape)
print("Shape of X_audiences:", X_audiences.shape)
print("Shape of y_ctr:", y_ctr.shape)






# # Convert and pad objective embeddings
# max_objective_length = max(len(eval(embed)) for embed in creatives_df['objective_embedding'])
# X_objectives1 = np.array([eval(embed) + [0] * (max_objective_length - len(eval(embed))) for embed in creatives_df['objective_embedding']])
# X_objectives=layers.Dense(128, activation=None)(X_objectives1)


# # Convert and pad objective embeddings
# max_goal_length = max(len(eval(embed)) for embed in creatives_df['goal_embedding'])
# X_goal1 = np.array([eval(embed) + [0] * (max_objective_length - len(eval(embed))) for embed in creatives_df['goal_embedding']])
# X_goal=layers.Dense(128, activation=None)(X_goal1)
# X_creatives_combined = np.concatenate([X_creatives, X_objectives, X_goal], axis=1)

# X_combined now contains the concatenated embeddings of creatives, objectives, and goals
# print(X_creatives_combined.shape)

print(X_creatives[0])

"""Two-Tower Model Implementation"""

import random
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Step 1: Data Preparation
# creatives_embeddings_dim=664
creatives_embeddings_dim=1024 #1408 for google multimodal embeddings
audience_embeddings_dim=768



# Step 2: Define Two-Tower Model
def create_creative_tower(embeddings_dim):
    creative_input = layers.Input(shape=(embeddings_dim,), name='creative_input')
    # x = layers.Dense(1408, activation='relu')(creative_input)
    x = layers.Dense(704, activation='relu')(creative_input)
    x = layers.Dense(216, activation='relu')(x)
    creative_output = layers.Dense(128, activation=None)(x) # Change output dimension to 32
    return Model(inputs=creative_input, outputs=creative_output, name='creative_tower')

def create_audience_tower(embeddings_dim):
    audience_input = layers.Input(shape=(embeddings_dim,), name='audience_input') #
    y = layers.Dense(532, activation='relu')(audience_input)
    y = layers.Dense(256, activation='relu')(y)
    audience_output = layers.Dense(128, activation=None)(y)
    return Model(inputs=audience_input, outputs=audience_output, name='audience_tower')

# Instantiate towers
creative_tower = create_creative_tower(creatives_embeddings_dim)
audience_tower = create_audience_tower(audience_embeddings_dim) # No need to pass embeddings_dim here as it's hardcoded in the function now


# Combine towers
creative_output = creative_tower.output
audience_output = audience_tower.output
similarity = layers.Dot(axes=-1, normalize=True)([creative_output, audience_output])
model = Model(inputs=[creative_tower.input, audience_tower.input], outputs=similarity)

"""Training"""

# Set random seed for reproducibility
import random
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)



# Compile the model with binary cross-entropy loss since we have binary labels for CTR
model.compile(optimizer=AdamW(), loss='binary_crossentropy', metrics=['accuracy'])  # Changed loss function to binary cross-entropy
# # Compile the model with mean squared error loss for regression
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])



# Import ModelCheckpoint callback
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the checkpoint to save the model with the lowest validation loss
checkpoint = ModelCheckpoint(
    'best_model.keras',  # Name of the file to save the model to
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)


# Train the model
from sklearn.model_selection import train_test_split


# Split data
X_creative_train, X_creative_val, X_audience_train, X_audience_val, y_train, y_val = train_test_split(
    X_creatives, X_audiences, y_ctr, test_size=0.2, random_state=42
)


# Training the model with the checkpoint callback
history = model.fit(
    [X_creative_train, X_audience_train], y_train,
    validation_data=([X_creative_val, X_audience_val], y_val),
    epochs=40,
    batch_size=16,
    callbacks=[checkpoint]  # Add the checkpoint callback here
)


# Extract trained embeddings after training
creative_embeddings = creative_tower.predict(X_creatives)
audience_embeddings = audience_tower.predict(X_audiences)

# Load the saved model
best_model = tf.keras.models.load_model('best_model.keras')

# Evaluate the loaded model on the validation set
val_loss, val_accuracy = best_model.evaluate([X_creative_val, X_audience_val], y_val)

print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

"""# Loading Interest Universe"""

# Load interest embeddings
interest_embeddings_df = pd.read_excel('/content/final_concatenated_universe_interests_converted_embed.xlsx')

import re

# Function to clean and convert space-separated strings to lists of floats
def convert_and_clean_embedding(embedding_str):
    # Remove any extraneous characters like brackets
    cleaned_str = re.sub(r'[\[\]\']', '', embedding_str)
    # Convert cleaned string to list of floats, splitting on commas
    return list(map(float, cleaned_str.split(',')))  # Split on commas here

# Apply the conversion function
interest_embeddings_df['embedding'] = interest_embeddings_df['embedding'].apply(convert_and_clean_embedding)

# Find the maximum length of embeddings
max_length = max(len(embed) for embed in interest_embeddings_df['embedding'])

# Pad embeddings to ensure all have the same length
padded_embeddings = [embed + [0] * (max_length - len(embed)) for embed in interest_embeddings_df['embedding']]

# Convert to numpy array
interest_embeddings = np.array(padded_embeddings)
np.save('interest_embeddings.npy', interest_embeddings)
# Example of how to use the embeddings
print(interest_embeddings[:5])
print(interest_embeddings.shape)

"""Passing the audience dataset through audience tower"""

interest_names = interest_embeddings_df['name'].tolist()

# Convert to numpy array
X_all_interests = np.array(interest_embeddings)

# Load the saved model
best_model = tf.keras.models.load_model('best_model.keras', custom_objects={'ModelCheckpoint': ModelCheckpoint})

interest_names = interest_embeddings_df['name'].tolist()
# Convert to numpy array
X_all_interests = np.array(interest_embeddings)

# Pass the reshaped embeddings through the audience tower
audience_embeddings_transformed = audience_tower.predict(X_all_interests)
print(audience_embeddings_transformed.shape)
audience_embeddings_transformed[26]

creative_embed_tran= creative_tower.predict(X_creatives)

print(audience_embeddings_transformed[:5])

print(creative_embed_tran[:5]) # print the first 5 rows of the array

"""# 2. Predictions using Scann"""

!pip install scann

import scann
# Get the list of interest IDs
interest_ids = interest_embeddings_df['id'].tolist()

# Define the SCANN searcher for the audience embeddings
searcher = scann.scann_ops_pybind.builder(audience_embeddings_transformed, 50, "dot_product") \
    .tree(num_leaves=168, num_leaves_to_search=100, training_sample_size=200).score_ah(2, anisotropic_quantization_threshold=0.2) \
    .reorder(100) \
    .build()

# Prepare the creative embedding
creative_embedding = X_creatives[4]
creative_embedding = np.expand_dims(creative_embedding, axis=0).astype(np.float32)  # Assuming your model expects float32
creative_embedding_transformed = creative_tower.predict(creative_embedding)

# Search for the top 20 most similar interest embeddings
indices, distances = searcher.search(creative_embedding_transformed[0])

# Output the top indices, distances, interest names, and corresponding ids
print("Top 50 most similar interests for the given creative:")
for idx, distance in zip(indices, distances):
    interest_name = interest_names[idx]
    # Retrieve the interest ID using the index from the interest_ids list
    interest_id = interest_ids[idx]
    print(f"Interest ID: {interest_id}, Interest Name: {interest_name}, Distance: {distance}")
