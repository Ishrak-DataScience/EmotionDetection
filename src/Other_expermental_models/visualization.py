# -*- coding: utf-8 -*-
"""Untitled27.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GVEvdT7f0xc_ZaTiJSZWFG8h1AZudc02
"""

import os

# Define the folder path
folder_path = '/content/drive/MyDrive/TUD Master/LLM/track_a/train'

# List to store the file names
csv_files = []

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a CSV file
    if file_name.endswith('.csv'):
        csv_files.append(file_name)

# Print the list of CSV file names
print("List of CSV files:")
for file_name in csv_files:
    print(file_name)

import os
import pandas as pd

folder_path = "/content/drive/MyDrive/TUD Master/LLM/track_a/train"
file_names = [
    'eng.csv', 'amh.csv', 'sun.csv', 'chn.csv', 'mar.csv', 'tir.csv',
    'esp.csv', 'ibo.csv', 'ukr.csv', 'hin.csv', 'deu.csv', 'som.csv',
    'swe.csv', 'kin.csv', 'vmw.csv', 'ptbr.csv', 'orm.csv', 'ary.csv',
    'pcm.csv', 'yor.csv', 'hau.csv', 'afr.csv', 'arq.csv', 'swa.csv',
    'tat.csv', 'rus.csv', 'ptmz.csv', 'ron.csv'
]
emotion_columns = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Disgust']

all_emotion_data = []

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    try:
        df = pd.read_csv(file_path)
        # Handle potential missing columns and case inconsistencies gracefully
        for col in emotion_columns:
            if col.lower() not in [c.lower() for c in df.columns]:
                df[col] = 0  # Add missing columns with default value 0
            else:
              # Find the correct column name, handling case differences
              correct_col = next((c for c in df.columns if c.lower() == col.lower()), None)
              if correct_col and correct_col != col:
                df.rename(columns={correct_col: col}, inplace=True)

        emotion_counts = df[emotion_columns].sum()
        emotion_data = {'Language': file_name.split('.')[0]}
        emotion_data.update(emotion_counts.to_dict())
        all_emotion_data.append(emotion_data)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")

# Create the final DataFrame
emotion_summary_df = pd.DataFrame(all_emotion_data)
emotion_summary_df = emotion_summary_df.set_index('Language')
emotion_summary_df

import os
import pandas as pd

folder_path = "/content/drive/MyDrive/TUD Master/LLM/track_a/train"
file_names = [
    'eng.csv', 'amh.csv', 'sun.csv', 'chn.csv', 'mar.csv', 'tir.csv',
    'esp.csv', 'ibo.csv', 'ukr.csv', 'hin.csv', 'deu.csv', 'som.csv',
    'swe.csv', 'kin.csv', 'vmw.csv', 'ptbr.csv', 'orm.csv', 'ary.csv',
    'pcm.csv', 'yor.csv', 'hau.csv', 'afr.csv', 'arq.csv', 'swa.csv',
    'tat.csv', 'rus.csv', 'ptmz.csv', 'ron.csv'
]
emotion_columns = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Disgust']

all_emotion_data = []

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    try:
        df = pd.read_csv(file_path)
        # Handle potential missing columns and case inconsistencies gracefully
        for col in emotion_columns:
            if col.lower() not in [c.lower() for c in df.columns]:
                df[col] = 0  # Add missing columns with default value 0
            else:
                # Find the correct column name, handling case differences
                correct_col = next((c for c in df.columns if c.lower() == col.lower()), None)
                if correct_col and correct_col != col:
                    df.rename(columns={correct_col: col}, inplace=True)

        # Compute the "Neutral" column
        df['Neutral'] = (df[emotion_columns].sum(axis=1) == 0).astype(int)

        # Summarize emotions
        emotion_counts = df[emotion_columns + ['Neutral']].sum()
        emotion_data = {'Language': file_name.split('.')[0]}
        emotion_data.update(emotion_counts.to_dict())
        all_emotion_data.append(emotion_data)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")

# Create the final DataFrame
emotion_summary_df = pd.DataFrame(all_emotion_data)
emotion_summary_df = emotion_summary_df.set_index('Language')

emotion_summary_df

# prompt: Using dataframe emotion_summary_df: visualize that table

import altair as alt

# Reset the index to make 'Language' a column again
emotion_summary_df = emotion_summary_df.reset_index()

# Melt the dataframe to long format for easier plotting
emotion_summary_long = emotion_summary_df.melt(id_vars='Language', var_name='Emotion', value_name='Value')

# Create the bar chart
chart = alt.Chart(emotion_summary_long).mark_bar().encode(
    x='Language',
    y='Value',
    color='Emotion'
).properties(
    width=600,
    height=400,
    title='Emotions by Language'
)

chart

# Calculate the total counts for each emotion across all languages
total_emotion_counts = emotion_summary_df.sum()

# Convert the total counts to a DataFrame for easier readability
total_emotion_df = total_emotion_counts.reset_index()
total_emotion_df.columns = ['Emotion', 'Total Count']

# Print the total counts
total_emotion_df

# List of required emotion columns (standardized to lowercase)
required_emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust']

# Initialize an empty dictionary to store the DataFrames
dataframes = {}  # This line is added

# Process all files to add missing emotion columns
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    lang = file_name.split('.')[0]  # Extract language from file name
    try:
        # Load the file into a DataFrame
        df = pd.read_csv(file_path)

        # Normalize column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]

        # Add missing emotion columns with default value 0
        for emotion in required_emotions:
            if emotion not in df.columns:
                df[emotion] = 0  # Add the missing emotion column with default value 0

        # Save the updated DataFrame back to the dictionary
        dataframes[lang] = df

        print(f"Processed {lang}: All emotion columns present.")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Debug: Confirm all DataFrames now have the required emotion columns
for lang, df in dataframes.items():
    print(f"Language: {lang}, Columns: {df.columns.tolist()}")

import os
import pandas as pd

# Folder path where the files are stored
folder_path = "/content/drive/MyDrive/TUD Master/LLM/track_a/train"

# List of file names and corresponding readable language names
file_names = {
    'eng.csv': 'English', 'amh.csv': 'Amharic', 'sun.csv': 'Sundanese', 'chn.csv': 'Chinese',
    'mar.csv': 'Marathi', 'tir.csv': 'Tigrinya', 'esp.csv': 'Spanish', 'ibo.csv': 'Igbo',
    'ukr.csv': 'Ukrainian', 'hin.csv': 'Hindi', 'deu.csv': 'German', 'som.csv': 'Somali',
    'swe.csv': 'Swedish', 'kin.csv': 'Kinyarwanda', 'vmw.csv': 'Makhuwa', 'ptbr.csv': 'Portuguese (Brazil)',
    'orm.csv': 'Oromo', 'ary.csv': 'Arabic (Moroccan)', 'pcm.csv': 'Nigerian Pidgin', 'yor.csv': 'Yoruba',
    'hau.csv': 'Hausa', 'afr.csv': 'Afrikaans', 'arq.csv': 'Arabic (Algerian)', 'swa.csv': 'Swahili',
    'tat.csv': 'Tatar', 'rus.csv': 'Russian', 'ptmz.csv': 'Portuguese (Mozambique)', 'ron.csv': 'Romanian'
}

# Standardized list of emotion labels (in lowercase)
required_emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust']

# Initialize a list to store emotion data for all languages
all_emotion_data = []

# Process all files
for file_name, readable_name in file_names.items():
    file_path = os.path.join(folder_path, file_name)
    lang = file_name.split('.')[0]  # Extract language code from file name
    try:
        # Load the file into a DataFrame
        df = pd.read_csv(file_path)

        # Normalize column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]

        # Ensure all required emotion columns are present
        for emotion in required_emotions:
            if emotion not in df.columns:
                df[emotion] = 0  # Add missing emotion column with default value 0

        # Sum the emotion counts
        emotion_counts = df[required_emotions].sum()

        # Prepare the data for this language
        emotion_data = {'Language Code': lang, 'Readable Language': readable_name}
        emotion_data.update(emotion_counts.to_dict())  # Add emotion counts to the dictionary

        # Append the result to the list
        all_emotion_data.append(emotion_data)

        print(f"Processed {readable_name}: All emotion columns ensured.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Create the final DataFrame
emotion_summary_df = pd.DataFrame(all_emotion_data)

# Output the result
emotion_summary_df

