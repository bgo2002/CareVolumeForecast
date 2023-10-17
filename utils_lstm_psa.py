import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def custom_aggregate(x):
    # Filter data where any of the VC columns is 1
    vc_data = x[(x['ms_teams'] == 1) | (x['phone'] == 1) | (x['telehealth'] == 1)]

    # Count unique patient and provider IDs for virtual care
    pt_vc_nb = vc_data['cleaned_patient_id'].nunique()
    pr_vc_nb = vc_data['cleaned_provider_id'].nunique()

    # Count total unique patient and provider IDs
    pt_nb = x['cleaned_patient_id'].nunique()
    pr_nb = x['cleaned_provider_id'].nunique()

    return pd.Series([pt_vc_nb, pr_vc_nb, pt_nb, pr_nb], index=['pt_vc_nb', 'pr_vc_nb', 'pt_nb', 'pr_nb'])

def aggregate_data(data, freq):
    # Resample and sum for appointment types
    resampled_data = data[['face_to_face', 'ms_teams', 'phone', 'telehealth']].resample(freq).sum()

    # Resample and apply custom aggregation for unique counts
    unique_counts = data.resample(freq).apply(custom_aggregate)

    # Create aggregated_data from resampled_data
    aggregated_data = resampled_data.copy()
    aggregated_data.loc[:, 'total'] = resampled_data.sum(axis=1)
    aggregated_data.loc[:, 'vc']    = resampled_data[['ms_teams', 'phone', 'telehealth']].sum(axis=1)
    aggregated_data.loc[:, 'vc_prop'] = aggregated_data.loc[:, 'vc'] / aggregated_data.loc[:, 'total']

    # Merge the unique counts with the aggregated_data
    aggregated_data = pd.concat([aggregated_data, unique_counts], axis=1)

    return aggregated_data

def plot_care_modalities(data, title_suffix=''):
    plt.figure(figsize=(10,6))
    plt.plot(data['face_to_face'], label='Face to Face')
    plt.plot(data['ms_teams'], label='MS Teams')
    plt.plot(data['phone'], label='Phone')
    plt.plot(data['telehealth'], label='Telehealth')
    plt.plot(data['vc'], label='VCs (ms/ph/te)')
    plt.plot(data['total'], label='Total')
    plt.title(f'Volume of Each Care Modality Over Time ({title_suffix})')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.legend(title="Care Modalities", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()

def plot_volumes_over_time(data, title_suffix=''):
    plt.figure(figsize=(10,6))
    plt.plot(data['pt_vc_nb'], label='Patients w/ VC')
    plt.plot(data['pt_nb'], label='Total Patients')
    plt.plot(data['pr_vc_nb'], label='Providers w/ VC')
    plt.plot(data['pr_nb'], label='Total Providers')
    plt.plot(data['vc'], label='VCs (ms/ph/te)')
    plt.plot(data['total'], label='Total Services')
    plt.title(f'Volumes Over Time ({title_suffix})')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.legend(title="Care Modalities", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()

def bar_plot(data, columns_to_plot, title):
    ax = data.iloc[:, columns_to_plot].plot(kind='bar', stacked=True, figsize=(10,6))

    labels = [item.get_text() for item in ax.get_xticklabels()]

    # Replace undesired labels with empty strings
    for i in range(len(labels)):
        if i % 10 != 0:  # Adjust this value to control the frequency of labels
            labels[i] = ''

    # Update tick labels
    ax.set_xticklabels(labels)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.legend(title="Care Modalities")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis tick labels by 45 degrees
    ax.legend(title="Care Modalities", loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

def create_event_dataframe(events, timestamps, start_date, end_date):
    
    # Convert event names to python-friendly format
    events = [event.lower().replace(" ", "_").replace("-", "_") for event in events]

    # Convert timestamps to datetime format
    timestamps = [pd.to_datetime(timestamp) for timestamp in timestamps]

    # Create a DataFrame
    psa = pd.DataFrame({
        'eventname': events,
        'timestamp': timestamps
    })

    # Filter events based on the given date range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    valid_events = psa[(psa['timestamp'] > start_date) & (psa['timestamp'] < end_date)]
    
    return valid_events


def encode_and_resample(valid_events, start_date, end_date, frequency='W-SUN'):

    # Create a date range from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date)

    # Initialize a new DataFrame with the date range as the index
    encoded_psa = pd.DataFrame(index=date_range)

    # For each event in valid_events, set the value to 1 from its date until the end_date
    for index, row in valid_events.iterrows():
        encoded_psa[row['eventname']] = (encoded_psa.index >= row['timestamp']).astype(int)

    # Fill NaN values with 0
    encoded_psa.fillna(0, inplace=True)

    # Resample the data based on the specified frequency and take the maximum value for each event
    resampled_psa = encoded_psa.resample(frequency).max()

    return resampled_psa


def get_features_and_target(data, feature_columns, target_column):
  
    features = data[feature_columns]
    target = data[[target_column]]
    
    return features, target
