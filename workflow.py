from truefoundry.workflow import task, workflow, PythonTaskConfig, TaskPythonBuild
from qdrant_client import QdrantClient
import numpy as np
import pandas as pd
import tensorflow as tf
from truefoundry.deploy import Resources
from google.cloud import bigquery, storage, aiplatform
import os
import requests
from moviepy.editor import VideoFileClip
import re
import json
import time
from tensorflow import keras
from pathlib import Path
from google.cloud.storage import Client, transfer_manager
from google.api_core.exceptions import ServiceUnavailable
from io import BytesIO
from typing import List
from tensorflow.keras import layers, Model
import ast
from tensorflow.keras.optimizers import AdamW
import random
from truefoundry.ml import get_client, ModelFramework
from tensorflow.keras.layers import TFSMLayer
from tqdm import tqdm






# Set up the task configuration
cpu_task_config = PythonTaskConfig(
    image=TaskPythonBuild(
        python_version="3.9",
        pip_packages=[
            "truefoundry[workflow]", "pandas", "google-cloud-bigquery",
            "openpyxl", "google-cloud-storage", "google-cloud-aiplatform",
            "moviepy", "numpy", "qdrant-client", "tensorflow","db-dtypes", "scikit-learn", "truefoundry[workflow]"

        ]
    ),
    resources=Resources(cpu_request=0.45, memory_request=1000, memory_limit=1000),
    service_account="flytepropeller",
)

@task(task_config=cpu_task_config)
def fetch_and_preprocess_data(platform_id: str, kpi_name: str, name: str) -> pd.DataFrame:
    client = bigquery.Client(project='the-story-teller-379307')
    table_name = f"{name}"

    # Fetch KPI information
    kpi_query = f"""
    SELECT kpi, columns, aggregationlogic, flag
    FROM `the-story-teller-379307.staging_storyteller.kpis`
    WHERE kpi = '{kpi_name}'
    """
    kpi_data = client.query(kpi_query).to_dataframe()

    # Set aggregation logic based on KPI
    if kpi_name == "costPerPurchase":
        kpi_columns_str = "spendInOrgCurrency, actionsOmniPurchase"
        agg_logic = "SAFE_DIVIDE(SUM(spendInOrgCurrency), SUM(actionsOmniPurchase))"
        flag = "Decreasing"
    else:
        kpi_columns_str = kpi_data['columns'].iloc[0]
        agg_logic = kpi_data['aggregationlogic'].iloc[0]
        flag = kpi_data['flag'].iloc[0]

    # Split KPI columns and set all column names
    kpi_columns_list = [col.strip() for col in kpi_columns_str.split(',')]
    fixed_columns = ['platformID', 'mediaAssetId', 'mediaAssetUrl', 'includedInterests']
    all_columns = fixed_columns + kpi_columns_list

    # Updated query to keep each includedInterests combination for a mediaAssetId
    query = f"""
    WITH performance AS (
        SELECT {', '.join(all_columns)}
        FROM `the-story-teller-379307.NIHKIL_DEV.{table_name}`
        WHERE platformID = '{platform_id}'
        AND NOT (mediaAssetId IS NULL OR platformID IS NULL OR mediaAssetUrl IS NULL OR includedInterests IS NULL
                 OR {' OR '.join([f'{col} IS NULL' for col in kpi_columns_list]) })
        AND NOT (contains_substr(includedInterests, 'behaviors') OR
                 contains_substr(includedInterests, 'life_events') OR
                 contains_substr(includedInterests, 'work') OR
                 contains_substr(includedInterests, 'education') OR
                 contains_substr(includedInterests, 'industries') OR
                 contains_substr(includedInterests, 'work_positions') OR
                 contains_substr(includedInterests, 'work_employers') OR
                 contains_substr(includedInterests, 'family_statuses') OR
                 contains_substr(includedInterests, 'education_majors'))
    ),
    aggregated_performance AS (
        SELECT mediaAssetId,
               ANY_VALUE(platformID) AS platformID,
               ANY_VALUE(mediaAssetUrl) AS mediaAssetUrl,
               includedInterests,
               {', '.join([f'SUM({col}) AS {col}' for col in kpi_columns_list])},
               CASE
                   WHEN '{agg_logic}' LIKE 'SAFE_DIVIDE%' THEN {agg_logic}
                   ELSE {agg_logic}
               END AS kpi,
               '{flag}' AS flag
        FROM performance
        GROUP BY mediaAssetId, includedInterests  -- Group by both mediaAssetId and includedInterests
        HAVING kpi IS NOT NULL
    ),
    global_kpi AS (
        SELECT
            CASE
                WHEN '{agg_logic}' LIKE 'SAFE_DIVIDE%' THEN {agg_logic}
                ELSE AVG(kpi)
            END AS global_kpi
        FROM aggregated_performance
    )
    SELECT p.*,
           g.global_kpi,
           CASE
               WHEN p.flag = 'Increasing' AND p.kpi >= g.global_kpi THEN 1
               WHEN p.flag = 'Increasing' AND p.kpi < g.global_kpi THEN 0
               WHEN p.flag = 'Decreasing' AND p.kpi <= g.global_kpi THEN 1
               WHEN p.flag = 'Decreasing' AND p.kpi > g.global_kpi THEN 0
               ELSE 0
           END AS label
    FROM aggregated_performance p, global_kpi g
    """

    # Run the query and save the data
    processed_data = client.query(query).to_dataframe()
    processed_data.to_excel("preprocessed_data_with_labels.xlsx", index=False)
    return processed_data

# Task 2: Separate Creative and Audience data
@task(task_config=cpu_task_config)
def separate_creative_and_audience_data(processed_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    creative_data = processed_data[['mediaAssetId', 'mediaAssetUrl', 'label']]
    audience_data = processed_data[['platformID', 'includedInterests']]
    creative_data.to_excel("creative_tower_data.xlsx", index=False)
    audience_data.to_excel("audience_tower_data.xlsx", index=False)
    return creative_data, audience_data


# Task: Process media assets (download, upload to GCS, extract audio if needed)
@task(task_config=cpu_task_config)
def process_media_assets(creative_data: pd.DataFrame, org_id: str) -> pd.DataFrame:
    bucket_name = "storyteller-vertex-ai"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Define folder paths on Google Cloud Storage
    image_folder = f"{org_id}_images"
    video_folder = f"{org_id}_videos"
    audio_folder = f"{org_id}_audios"

    # Create folders in GCS if they don’t exist
    for folder in [image_folder, video_folder, audio_folder]:
        blob = bucket.blob(f"{folder}/")
        blob.upload_from_string('')

    # Helper function to download files locally
    def download_file(url, save_path):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return True
        return False

    # Helper function to upload files to GCS
    def upload_to_gcs(local_path, gcs_path):
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

    # Process each media asset in the creative data
    for idx, link in enumerate(creative_data['mediaAssetUrl']):
        file_name = os.path.basename(link)
        ext = file_name.split('.')[-1].lower()

        # Determine if the file is an image or video
        if ext == 'jpg':  # Image format
            local_path = os.path.join('/tmp', file_name)
            if download_file(link, local_path):
                gcs_path = f"{image_folder}/{file_name}"
                upload_to_gcs(local_path, gcs_path)

        elif ext == 'mp4':  # Video format
            local_video_path = os.path.join('/tmp', file_name)
            if download_file(link, local_video_path):
                gcs_video_path = f"{video_folder}/{file_name}"
                upload_to_gcs(local_video_path, gcs_video_path)

                # Extract audio from video if it exists
                video_clip = VideoFileClip(local_video_path)
                if video_clip.audio is not None:
                    local_audio_path = os.path.join('/tmp', f"{os.path.splitext(file_name)[0]}.wav")
                    video_clip.audio.write_audiofile(local_audio_path)

                    # Upload extracted audio to GCS
                    gcs_audio_path = f"{audio_folder}/{os.path.splitext(file_name)[0]}.wav"
                    upload_to_gcs(local_audio_path, gcs_audio_path)
                video_clip.close()

    return creative_data

from time import sleep

# Task: Retrieve multimodal embeddings from the ImageBind endpoint
@task(task_config=cpu_task_config)
def retrieve_multimodal_embeddings(creative_data: pd.DataFrame, org_id: str) -> pd.DataFrame:
    bucket_name = "storyteller-vertex-ai"
    image_folder = f"{org_id}_images"
    video_folder = f"{org_id}_videos"
    audio_folder = f"{org_id}_audios"
    endpoint = aiplatform.Endpoint('projects/the-story-teller-379307/locations/us-central1/endpoints/5218195324006301696')
    storage_client = storage.Client()

    # Helper function to fetch embeddings for a list of files in a specific modality
    def get_embeddings(file_paths, prefix, modality):
        embeddings = {}
        batch_size = 3  # Reduced batch size to handle smaller requests
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            instances = [{modality: [f"gs://{bucket_name}/{prefix}/{file}" for file in batch_paths]}]
            retries = 3
            for attempt in range(retries):
                try:
                    print(f"Requesting embeddings for {modality}, files: {batch_paths}")
                    response = endpoint.predict(instances=instances)
                    if response and response.predictions:
                        for idx, file_path in enumerate(batch_paths):
                            embedding_array = np.array(response.predictions[0][modality][idx])
                            embedding_str = "[" + ", ".join(map(str, embedding_array)) + "]"
                            embeddings[os.path.basename(file_path)] = embedding_str
                    else:
                        print(f"No embeddings returned for {file_paths}")
                    break
                except Exception as e:
                    print(f"Error retrieving embeddings for {batch_paths}: {e}")
                    if attempt < retries - 1:
                        print(f"Retrying in {2 ** attempt} seconds...")
                        sleep(2 ** attempt)  # Exponential backoff
                    else:
                        print(f"Failed to retrieve embeddings for {batch_paths} after {retries} attempts.")
        return embeddings

    # Retrieve the list of files in GCS for each media type
    image_files = [blob.name.split('/')[-1] for blob in storage_client.bucket(bucket_name).list_blobs(prefix=image_folder) if blob.name.endswith('.jpg')]
    video_files = [blob.name.split('/')[-1] for blob in storage_client.bucket(bucket_name).list_blobs(prefix=video_folder) if blob.name.endswith('.mp4')]
    audio_files = [blob.name.split('/')[-1] for blob in storage_client.bucket(bucket_name).list_blobs(prefix=audio_folder) if blob.name.endswith('.wav')]

    # Fetch embeddings for each media type
    image_embeddings = get_embeddings(image_files, image_folder, 'vision')
    video_embeddings = get_embeddings(video_files, video_folder, 'video')
    audio_embeddings = get_embeddings(audio_files, audio_folder, 'audio')

    # Calculate averaged embeddings for each media asset in creative_data
    def average_embeddings(media_asset_url):
        filename = os.path.splitext(os.path.basename(media_asset_url))[0]
        embeddings_list = []
        image_embedding_str = image_embeddings.get(f"{filename}.jpg")
        video_embedding_str = video_embeddings.get(f"{filename}.mp4")
        audio_embedding_str = audio_embeddings.get(f"{filename}.wav")

        if image_embedding_str:
            image_embedding = np.array([float(x) for x in image_embedding_str.strip("[]").split(',')])
            embeddings_list.append(image_embedding)
        if video_embedding_str:
            video_embedding = np.array([float(x) for x in video_embedding_str.strip("[]").split(',')])
            embeddings_list.append(video_embedding)
        if audio_embedding_str:
            audio_embedding = np.array([float(x) for x in audio_embedding_str.strip("[]").split(',')])
            embeddings_list.append(audio_embedding)

        if embeddings_list:
            averaged_embedding = np.mean(embeddings_list, axis=0)
        else:
            averaged_embedding = []

        return "[" + ", ".join(map(str, averaged_embedding)) + "]"

    creative_data['embeddings'] = creative_data['mediaAssetUrl'].apply(average_embeddings)
    return creative_data









# Task: Convert audience JSON to English statements for each row
@task(task_config=cpu_task_config)
def convert_json_to_statements(audience_data: pd.DataFrame) -> pd.DataFrame:
    def clean_json_string(json_str):
        try:
            clean_str = json_str.replace('\\"', '"').replace("\\", "")
            return json.loads(clean_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return []

    def targeting_to_english(targeting):
        statement = []
        for spec in targeting:
            parts = []
            for key, value in spec.items():
                if key in ['interests', 'behaviors', 'life_events']:
                    item_names = [f"{item['name']} (id: {item['id']})" for item in value]
                    parts.append(f"({' OR '.join(item_names)})")
            if parts:
                statement.append(" AND ".join(parts))
        return " AND ".join(statement)

    english_statements = []
    for _, row in audience_data.iterrows():
        targeting_spec = clean_json_string(row['includedInterests']) if pd.notna(row['includedInterests']) else []
        english_statement = targeting_to_english(targeting_spec)
        english_statements.append(english_statement)
    audience_data['Targeting English Statement'] = english_statements
    return audience_data

# Task: Sum embeddings for all rows, extracting embeddings from Qdrant as you go
@task(task_config=cpu_task_config)
def sum_embeddings(df_statements: pd.DataFrame) -> pd.DataFrame:
    client = QdrantClient(url="https://staging-vector-db-storyteller-staging.app-staging.thestoryteller.tech:443")

    def extract_ids(statement: str) -> List[int]:
        return [int(id_) for id_ in re.findall(r'id: (\d+)', statement)]

    def get_embedding_from_qdrant(id_: int) -> np.ndarray:
        try:
            result = client.retrieve(
                with_vectors=True,
                collection_name="universe_embeddings",
                ids=[id_]
            )
            if result and result[0].vector:
                embedding_data = result[0].vector['default']
                print(f"Embedding for ID {id_}: {embedding_data}")
                embedding_vector = np.array(embedding_data, dtype=np.float32)
                if embedding_vector.shape == (768,):
                    return embedding_vector
                else:
                    print(f"Invalid embedding shape for ID: {id_}, using zero vector.")
                    return np.zeros(768)
            else:
                print(f"No data or empty embedding found for ID: {id_}, using zero vector.")
                return np.zeros(768)
        except Exception as e:
            print(f"Error retrieving embedding for ID {id_}: {e}, using zero vector.")
            return np.zeros(768)

    summed_embeddings = []
    for statement in df_statements['Targeting English Statement']:
        ids = extract_ids(statement)
        sum_embedding = np.zeros(768)
        for id_ in ids:
            embedding = get_embedding_from_qdrant(id_)
            sum_embedding += embedding
        summed_embeddings.append(sum_embedding)

    df_statements['Summed Embedding'] = [' '.join(str(x) for x in arr) for arr in summed_embeddings]
    return df_statements



def parse_embedding(embedding_str):
    embedding_str = embedding_str.strip('[]')
    components = embedding_str.replace('\n', '').split()
    return [float(comp) for comp in components if comp.strip()]

# Task 4: Data Preparation for Training
@task(task_config=cpu_task_config)
def prepare_data(creative_data: pd.DataFrame, audience_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Fill missing embeddings with empty lists
    creative_data['embeddings'] = creative_data['embeddings'].fillna('[]')

    # Find the maximum length of embeddings for padding
    max_length = max(len(eval(embed)) for embed in creative_data['embeddings'])

    # Convert the creative embeddings into padded NumPy arrays
    X_creatives = np.array([eval(embed) + [0] * (max_length - len(eval(embed))) for embed in creative_data['embeddings']])

    # Convert the audience embeddings into padded NumPy arrays
    def pad_embeddings(df):
        X_audiences_list = [parse_embedding(embed) for embed in df['Summed Embedding']]
        max_length = max(len(embed) for embed in X_audiences_list)
        X_audiences_padded = [list(embed) + [0] * (max_length - len(embed)) for embed in X_audiences_list]
        return np.array(X_audiences_padded)

    X_audiences = pad_embeddings(audience_data)

    # Convert labels into NumPy arrays
    y_label = creative_data['label'].to_numpy()

    return X_creatives, X_audiences, y_label



@task(task_config=cpu_task_config)
def define_train_and_save_model(X_creatives: np.ndarray, X_audiences: np.ndarray, y_label: np.ndarray, org_id: str, platform_id: str) -> tuple[str, str]:
    from tensorflow.keras import layers, Model
    from tensorflow.keras.optimizers import AdamW
    from tensorflow.keras.callbacks import ModelCheckpoint
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    creatives_embeddings_dim = X_creatives.shape[1]
    audience_embeddings_dim = X_audiences.shape[1]

    # Define the creative and audience towers
    def create_creative_tower(embeddings_dim):
        creative_input = layers.Input(shape=(embeddings_dim,), name='creative_input')
        x = layers.Dense(704, activation='relu')(creative_input)
        x = layers.Dense(216, activation='relu')(x)
        creative_output = layers.Dense(128, activation=None)(x)
        return Model(inputs=creative_input, outputs=creative_output, name='creative_tower')

    def create_audience_tower(embeddings_dim):
        audience_input = layers.Input(shape=(embeddings_dim,), name='audience_input')
        y = layers.Dense(532, activation='relu')(audience_input)
        y = layers.Dense(256, activation='relu')(y)
        audience_output = layers.Dense(128, activation=None)(y)
        return Model(inputs=audience_input, outputs=audience_output, name='audience_tower')

    creative_tower = create_creative_tower(creatives_embeddings_dim)
    audience_tower = create_audience_tower(audience_embeddings_dim)

    # Merge the two towers by calculating similarity (dot product)
    creative_output = creative_tower.output
    audience_output = audience_tower.output
    similarity = layers.Dot(axes=-1, normalize=True)([creative_output, audience_output])

    # Final model
    model = Model(inputs=[creative_tower.input, audience_tower.input], outputs=similarity, name="two_tower_model")
    model.compile(optimizer=AdamW(), loss='binary_crossentropy', metrics=['accuracy'])

    # Split the data into training and validation sets
    X_creative_train, X_creative_val, X_audience_train, X_audience_val, y_train, y_val = train_test_split(
        X_creatives, X_audiences, y_label, test_size=0.2, random_state=42
    )

    # Add model checkpoint callback
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    # Train the model
    model.fit(
        [X_creative_train, X_audience_train], y_train,
        validation_data=([X_creative_val, X_audience_val], y_val),
        epochs=40, batch_size=16, callbacks=[checkpoint]
    )

    # Save the creative and audience towers
    creative_tower_path = f"{org_id}_{platform_id}_creative_tower_model"
    audience_tower_path = f"{org_id}_{platform_id}_audience_tower_model"

    tf.saved_model.save(creative_tower, creative_tower_path)
    tf.saved_model.save(audience_tower, audience_tower_path)

    # Upload the models to TrueFoundry model registry (example code)
    client = get_client()

    creative_model_version = client.log_model(
        ml_repo="two-tower-model",
        name=f"{org_id}_{platform_id}_creative_tower",
        model_file_or_folder=creative_tower_path,
        framework=ModelFramework.TENSORFLOW,
        metadata={"description": "Creative tower model of the two-tower system"}
    )

    audience_model_version = client.log_model(
        ml_repo="two-tower-model",
        name=f"{org_id}_{platform_id}_audience_tower",
        model_file_or_folder=audience_tower_path,
        framework=ModelFramework.TENSORFLOW,
        metadata={"description": "Audience tower model of the two-tower system"}
    )

    return creative_model_version.fqn, audience_model_version.fqn


@task(task_config=cpu_task_config)
def transform_and_store_audience_embeddings_in_qdrant(org_id: str, platform_id: str, kpi_name: str, audience_tower_fqn: str) -> str:
    print("Connecting to Qdrant database...")
    client = QdrantClient(url="https://staging-vector-db-storyteller-staging.app-staging.thestoryteller.tech:443")
    print("Connected to Qdrant database.")

    new_collection_name = f"{org_id}_{platform_id}_{kpi_name}_audience_data"

    # Download the model from TrueFoundry
    print("Downloading model from TrueFoundry...")
    tf_client = get_client()
    model_version = tf_client.get_model_version_by_fqn(audience_tower_fqn)
    model_download_path = "model_downloads1"
    os.makedirs(model_download_path, exist_ok=True)
    download_info = model_version.download(path=model_download_path, overwrite=True)
    audience_model_path = download_info.model_dir
    print(f"Model downloaded to path: {audience_model_path}")

    # Load the model using TFSMLayer
    print("Loading the model using TFSMLayer...")
    audience_tower = TFSMLayer(audience_model_path, call_endpoint='serving_default')
    print("Model loaded successfully.")

    # Create collection in Qdrant
    print(f"Creating collection {new_collection_name} in Qdrant...")
    client.recreate_collection(
        collection_name=new_collection_name,
        vectors_config={
            "size": 128,  # expected vector size
            "distance": "Dot"
        }
    )
    print(f"Collection {new_collection_name} created in Qdrant.")

    # Retrieve all data points in `universe_embeddings` using `scroll`
    print("Retrieving all embeddings from the universe_embeddings collection...")
    batch_size = 2000
    offset = None
    transformed_data = []

    print("Starting transformation and upload process...")
    while True:
        # Retrieve a batch of data points with `scroll`
        scroll_result, next_offset = client.scroll(
            collection_name="universe_embeddings",
            limit=batch_size,
            with_vectors=True,
            offset=offset
        )

        # If no more data is found, end the loop
        if not scroll_result:
            print("No more embeddings to retrieve.")
            break

        print(f"Batch retrieved with {len(scroll_result)} embeddings.")

        # Process each item in the batch
        for item in scroll_result:
            try:
                # Access item attributes
                item_id = item.id
                item_payload = item.payload

                # Extract vector from nested structure if it contains 'default' key
                item_vector = item.vector.get("default")

                # Validate item_vector
                if item_vector is None or not isinstance(item_vector, list) or not all(isinstance(v, (float, int)) for v in item_vector):
                    print(f"Skipping item due to invalid vector format: {item.vector}")
                    continue

            except AttributeError as e:
                print(f"Error accessing item attributes: {e}")
                continue

            # Convert the embedding to numpy and pass through the model
            embedding_vector = np.array(item_vector, dtype=np.float32)
            transformed_output = audience_tower(tf.convert_to_tensor([embedding_vector], dtype=tf.float32))

            # Extract the transformed embedding
            transformed_embedding = transformed_output['output_0'].numpy()[0]

            # Prepare the transformed point with metadata for uploading
            transformed_data.append({
                "id": int(item_id),
                "vector": transformed_embedding.tolist(),  # Converted to list for Qdrant compatibility
                "payload": item_payload  # Retain original metadata
            })

        # Upload the batch to Qdrant
        if transformed_data:
            print(f"Uploading batch of {len(transformed_data)} transformed embeddings...")
            client.upsert(collection_name=new_collection_name, points=transformed_data)
            transformed_data.clear()  # Clear batch after upload
            print("Batch upload completed.")

        # Update offset for the next scroll call
        offset = next_offset if next_offset else None

        if len(scroll_result) < batch_size:
            print("All embeddings retrieved and uploaded.")
            break

    print(f"Process completed: Total embeddings transformed and stored in Qdrant collection: {new_collection_name}")
    return new_collection_name

@workflow
def combined_workflow(platform_id: str, kpi_name: str, name: str, org_id: str) -> tuple[str, str, str]:
    # Step 1: Fetch and preprocess the data from BigQuery
    processed_data = fetch_and_preprocess_data(platform_id=platform_id, kpi_name=kpi_name, name=name)

    # Step 2: Separate creative and audience data
    creative_data, audience_data = separate_creative_and_audience_data(processed_data=processed_data)

    # Step 3: Download media assets and generate multimodal embeddings
    creative_data_with_embeddings = process_media_assets(creative_data=creative_data, org_id=org_id)
    creative_data_with_embeddings = retrieve_multimodal_embeddings(creative_data=creative_data_with_embeddings, org_id=org_id)

    # Step 4: Convert audience JSON to English statements
    audience_data_with_statements = convert_json_to_statements(audience_data=audience_data)

    # Step 5: Sum embeddings for audience data
    audience_data_with_embeddings = sum_embeddings(df_statements=audience_data_with_statements)

    # Step 6: Prepare data for training by padding embeddings
    X_creatives, X_audiences, y_label = prepare_data(
        creative_data=creative_data_with_embeddings, audience_data=audience_data_with_embeddings
    )

    # Step 7: Define, train, and save the two-tower model
    model_paths = define_train_and_save_model(
        X_creatives=X_creatives, X_audiences=X_audiences, y_label=y_label, org_id=org_id, platform_id=platform_id
    )

    creative_tower_fqn = model_paths[0]
    audience_tower_fqn = model_paths[1]

    # Step 8: Pass the universe audiences through the trained audience tower and store the transformed embeddings
    new_audience_collection_name = transform_and_store_audience_embeddings_in_qdrant(
        org_id=org_id, platform_id=platform_id, kpi_name=kpi_name, audience_tower_fqn=audience_tower_fqn
    )

    return creative_tower_fqn, audience_tower_fqn, new_audience_collection_name
