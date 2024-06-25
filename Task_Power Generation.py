#!/usr/bin/env python
# coding: utf-8

# # step : Imports & Reading Data

# In[1]:


pip install pandas sqlite3 matplotlib seaborn


# In[2]:


import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
pd.set_option('display.max_columns', None)


# In[3]:


# Connect to SQLite database
conn = sqlite3.connect('cruise_data.db')
cursor = conn.cursor()

# Load CSV data into a DataFrame
df = pd.read_csv('C:/Users/seoin/Desktop/Tui/task_data/data.csv')

# Create a table in the SQLite database
df.to_sql('cruise_data', conn, if_exists='replace', index=False)

# Verify the data is loaded
query = "SELECT * FROM cruise_data"
df_sql = pd.read_sql_query(query, conn)
print(df_sql.head())


# # Step : Data Understanding

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.head()


# In[7]:


df.columns


# # Step : Data Preperation

# In[8]:


# Convert time columns to datetime dtype
df['Start Time'] = pd.to_datetime(df['Start Time'])
df['End Time'] = pd.to_datetime(df['End Time'])

# Display info for only 'Start Time' and 'End Time' columns
df[['Start Time', 'End Time']].info()


# ## Missing value check

# In[9]:


df.isnull().sum()


# In[10]:


df.duplicated().sum()


# In[11]:


df[df['Power Galley 1 (MW)'].isna()]


# In[12]:


# Drop rows where 'Power Galley 1 (MW)' has missing values becasue most of the data values are empty
df = df.dropna(subset=['Power Galley 1 (MW)'])


# In[13]:


df[df['HVAC Chiller 1 Power (MW)'].isna()]


# In[14]:


# Ensure 'Start Time' is in datetime format
df['Start Time'] = pd.to_datetime(df['Start Time'])

# Plot the time series for 'HVAC Chiller 1 Power (MW)'
plt.figure(figsize=(14, 7))
plt.plot(df['Start Time'], df['HVAC Chiller 1 Power (MW)'], label='HVAC Chiller 1 Power (MW)', color='blue')
plt.xlabel('Time')
plt.ylabel('HVAC Chiller 1 Power (MW)')
plt.title('HVAC Chiller 1 Power (MW) Over Time')
plt.legend()
plt.grid(True)
plt.show()


# In[15]:


# Sort the dataframe by 'Start Time' to ensure proper filling
df = df.sort_values(by='Start Time')

# Set 'Start Time' as the index
df.set_index('Start Time', inplace=True)

# Interpolate to fill missing values for chiller columns
#interpolation is a process of determining the unknown values that lie in between the known data points
df['HVAC Chiller 1 Power (MW)'] = df['HVAC Chiller 1 Power (MW)'].interpolate()
df['HVAC Chiller 2 Power (MW)'] = df['HVAC Chiller 2 Power (MW)'].interpolate()
df['HVAC Chiller 3 Power (MW)'] = df['HVAC Chiller 3 Power (MW)'].interpolate()

# If you need to reset the index back to columns
df.reset_index(inplace=True)


# In[16]:


df[df['Power Service (MW)'].isna()]


# In[17]:


# First forward fill, then backward fill
# the next available value after the missing data point replaces the missing value because it is only two missing values
df['Power Service (MW)'] = df['Power Service (MW)'].ffill().bfill()

print(df[['Start Time', 'Power Service (MW)']].head())


# In[18]:


df[df['Speed Over Ground (knots)'].isna()]


# In[19]:


# Plot the time series for 'Speed Over Ground (knots)'
plt.figure(figsize=(14, 7))
plt.plot(df['Start Time'], df['Speed Over Ground (knots)'], label='Speed Over Ground (knots)', color='blue')
plt.xlabel('Time')
plt.ylabel('Speed Over Ground (knots)')
plt.title('Speed Over Ground (knots) Over Time')
plt.legend()
plt.grid(True)
plt.show()


# In[20]:


# Sort the dataframe by 'Start Time' to ensure proper filling
df = df.sort_values(by='Start Time')

# Set 'Start Time' as the index
df.set_index('Start Time', inplace=True)

# Interpolate to fill missing values
df['Speed Over Ground (knots)'] = df['Speed Over Ground (knots)'].interpolate()

# If you need to reset the index back to columns
df.reset_index(inplace=True)



# In[21]:


# # missing value check 
df.isnull().sum()


# In[22]:


# Set 'Start Time' as the index
df.set_index('Start Time', inplace=True)

# Interpolate to fill missing values
df['Speed Through Water (knots)'] = df['Speed Through Water (knots)'].interpolate()

# If you need to reset the index back to columns
df.reset_index(inplace=True)

# Plot the time series for 'Speed Through Water (knots)'
plt.figure(figsize=(14, 7))
plt.plot(df['Start Time'], df['Speed Through Water (knots)'], label='Speed Through Water (knots)', color='blue')
plt.xlabel('Time')
plt.ylabel('Speed Through Water (knots)')
plt.title('Speed Through Water (knots) Over Time')
plt.legend()
plt.grid(True)
plt.show()


# In[23]:


# Set 'Start Time' as the index
df.set_index('Start Time', inplace=True)

# Interpolate to fill missing values
df['Speed Through Water (knots)'] = df['Speed Through Water (knots)'].interpolate()

# If you need to reset the index back to columns
df.reset_index(inplace=True)

print(df[['Start Time', 'Speed Through Water (knots)']].head())


# In[24]:


df.isnull().sum()


# ## Sorting Vessels data to Vessel 1 & Vessel 2

# In[25]:


# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1']
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2']

# Verify the split
print(vessel_1_data.shape)
print(vessel_2_data.shape)


# In[26]:


df.columns


# ## Outlier check of Power Generation

# In[27]:


power_generation_columns = [
    'Diesel Generator 1 Power (MW)', 'Diesel Generator 2 Power (MW)', 
    'Diesel Generator 3 Power (MW)', 'Diesel Generator 4 Power (MW)',
    'HVAC Chiller 1 Power (MW)', 'HVAC Chiller 2 Power (MW)', 
    'HVAC Chiller 3 Power (MW)', 'Scrubber Power (MW)', 
    'Power Galley 1 (MW)', 'Power Galley 2 (MW)', 'Power Service (MW)'
]

# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()


# In[28]:


# Function to create box plots for outliers visualization
def plot_outliers(df, columns, vessel_name):
    plt.figure(figsize=(20, 10))
    df[columns].boxplot()
    plt.title(f'Outlier Visualization for {vessel_name}', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.xticks(rotation=15, fontsize=20)  # Change rotation to 0 for horizontal labels
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to make room for the labels
    plt.show()

# Plot outliers for Vessel 1
plot_outliers(vessel_1_data, power_generation_columns, 'Vessel 1')

# Plot outliers for Vessel 2
plot_outliers(vessel_2_data, power_generation_columns, 'Vessel 2')


# ### Each outlier check

# In[29]:


# Ensure 'Start Time' is in datetime format and set as index
df['Start Time'] = pd.to_datetime(df['Start Time'])
vessel_1_data.set_index('Start Time', inplace=True)
vessel_2_data.set_index('Start Time', inplace=True)

# Function to detect outliers using IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Function to plot original data with outliers highlighted
def plot_with_outliers(df, column, outliers, vessel_name):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[column], label='Original Data')
    plt.scatter(outliers.index, outliers[column], color='red', label='Outliers')
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.title(f'{column} Over Time with Outliers Highlighted for {vessel_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to calculate and print statistical summaries
def print_summaries(df, column, outliers):
    data_without_outliers = df[~df.index.isin(outliers.index)]
    summary_with_outliers = df[column].describe()
    summary_without_outliers = data_without_outliers[column].describe()
    
    print(f"Summary with Outliers for {column}:\n", summary_with_outliers)
    print(f"\nSummary without Outliers for {column}:\n", summary_without_outliers)

# Columns to check
columns_to_check = ['Power Galley 1 (MW)', 'Power Galley 2 (MW)', 'Power Service (MW)']

# Vessel 1 Analysis
print("Vessel 1 Analysis:\n")
for column in columns_to_check:
    outliers_vessel_1 = detect_outliers(vessel_1_data, column)
    plot_with_outliers(vessel_1_data, column, outliers_vessel_1, 'Vessel 1')
    print_summaries(vessel_1_data, column, outliers_vessel_1)

# Vessel 2 Analysis
print("\nVessel 2 Analysis:\n")
for column in columns_to_check:
    outliers_vessel_2 = detect_outliers(vessel_2_data, column)
    plot_with_outliers(vessel_2_data, column, outliers_vessel_2, 'Vessel 2')
    print_summaries(vessel_2_data, column, outliers_vessel_2)


# Outliers Delete or keep it depends on the status

# In[30]:


# Ensure 'Start Time' is in datetime format and set as index
df['Start Time'] = pd.to_datetime(df['Start Time'])
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()
vessel_2_data.set_index('Start Time', inplace=True)

# Function to calculate Q1 and Q3
def calculate_q1_q3(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    return Q1, Q3

# Calculate Q1 and Q3 for the relevant columns
q1_pg1, q3_pg1 = calculate_q1_q3(vessel_2_data, 'Power Galley 1 (MW)')
q1_pg2, q3_pg2 = calculate_q1_q3(vessel_2_data, 'Power Galley 2 (MW)')
q1_ps, q3_ps = calculate_q1_q3(vessel_2_data, 'Power Service (MW)')

# Remove outliers
vessel_2_data_cleaned = vessel_2_data[
    (vessel_2_data['Power Galley 1 (MW)'] <= q3_pg1) &
    (vessel_2_data['Power Galley 2 (MW)'] <= q3_pg2) &
    (vessel_2_data['Power Service (MW)'] >= q1_ps)
]


# Plotting function to visualize the cleaned data (Optional)
def plot_cleaned_data(df, columns, vessel_name):
    plt.figure(figsize=(20, 10))
    for column in columns:
        plt.plot(df.index, df[column], label=column)
    plt.xlabel('Time')
    plt.ylabel('Power (MW)')
    plt.title(f'Power Generation Metrics Over Time for {vessel_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the cleaned data for the relevant columns
plot_cleaned_data(vessel_2_data_cleaned, ['Power Galley 1 (MW)', 'Power Galley 2 (MW)', 'Power Service (MW)'], 'Vessel 2')


# # Data Analysis

# In[31]:


# Define categories for Power Generation Analysis
diesel_generators = ['Diesel Generator 1 Power (MW)', 'Diesel Generator 2 Power (MW)', 
                     'Diesel Generator 3 Power (MW)', 'Diesel Generator 4 Power (MW)']
hvac_chillers = ['HVAC Chiller 1 Power (MW)', 'HVAC Chiller 2 Power (MW)', 'HVAC Chiller 3 Power (MW)']
scrubber_power = ['Scrubber Power (MW)']
power_galley = ['Power Galley 1 (MW)', 'Power Galley 2 (MW)']
power_service = ['Power Service (MW)']

# Combine all columns into a single list
power_columns = diesel_generators + hvac_chillers + scrubber_power + power_galley + power_service

# Ensure all columns are numeric
df[power_columns] = df[power_columns].apply(pd.to_numeric, errors='coerce')

# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()

# Calculate monthly averages for Vessel 1
vessel_1_data.set_index('Start Time', inplace=True)
vessel_1_monthly_avg = vessel_1_data[power_columns].resample('M').mean()

# Calculate monthly averages for Vessel 2
vessel_2_data.set_index('Start Time', inplace=True)
vessel_2_monthly_avg = vessel_2_data[power_columns].resample('M').mean()

# Function to plot monthly averages for Vessel 1
def plot_monthly_averages_vessel_1(columns, title):
    plt.figure(figsize=(14, 7))
    for column in columns:
        plt.plot(vessel_1_monthly_avg.index, vessel_1_monthly_avg[column], label=f'Vessel 1 {column}')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(f'Monthly Average {title} Over Time for Vessel 1')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot monthly averages for Vessel 2
def plot_monthly_averages_vessel_2(columns, title):
    plt.figure(figsize=(14, 7))
    for column in columns:
        plt.plot(vessel_2_monthly_avg.index, vessel_2_monthly_avg[column], label=f'Vessel 2 {column}')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(f'Monthly Average {title} Over Time for Vessel 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting for each category for Vessel 1
plot_monthly_averages_vessel_1(diesel_generators, 'Diesel Generator Power')
plot_monthly_averages_vessel_1(hvac_chillers, 'HVAC Chiller Power')
plot_monthly_averages_vessel_1(scrubber_power, 'Scrubber Power')
plot_monthly_averages_vessel_1(power_galley, 'Power Galley')
plot_monthly_averages_vessel_1(power_service, 'Power Service')

# Plotting for each category for Vessel 2
plot_monthly_averages_vessel_2(diesel_generators, 'Diesel Generator Power')
plot_monthly_averages_vessel_2(hvac_chillers, 'HVAC Chiller Power')
plot_monthly_averages_vessel_2(scrubber_power, 'Scrubber Power')
plot_monthly_averages_vessel_2(power_galley, 'Power Galley')
plot_monthly_averages_vessel_2(power_service, 'Power Service')


# In[32]:


# Combine all columns into a single list
power_columns = diesel_generators + hvac_chillers + scrubber_power + power_galley + power_service

# Ensure all columns are numeric
df[power_columns] = df[power_columns].apply(pd.to_numeric, errors='coerce')

# Ensure 'Start Time' is in datetime format and set as index
df['Start Time'] = pd.to_datetime(df['Start Time'])
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()
vessel_1_data.set_index('Start Time', inplace=True)
vessel_2_data.set_index('Start Time', inplace=True)

# Calculate monthly averages for Vessel 1
vessel_1_monthly_avg = vessel_1_data[power_columns].resample('M').mean()

# Calculate monthly averages for Vessel 2
vessel_2_monthly_avg = vessel_2_data[power_columns].resample('M').mean()

# Prepare DataFrames for SQL storage
power_vessel_1 = vessel_1_monthly_avg.copy()
power_vessel_1['Vessel'] = 'Vessel 1'

power_vessel_2 = vessel_2_monthly_avg.copy()
power_vessel_2['Vessel'] = 'Vessel 2'

# Combine all results
combined_power_results = pd.concat([power_vessel_1, power_vessel_2])

# Store combined results into SQLite database
combined_power_results.to_sql('power_generation_analysis', conn, if_exists='replace', index=True)

# Verify data is stored correctly
query = "SELECT * FROM power_generation_analysis"
df_sql = pd.read_sql_query(query, conn)
print(df_sql.head())

# Close the connection
conn.close()


# # Power Generation Performance Trend Analysis Report
# 
# **Introduction**
# 
# The power generation performance of vessels is critical to ensure operational efficiency and compliance with environmental standards. This report analyzes the trends in power generation for two vessels over a year. The analysis covers various components such as diesel generators, HVAC chillers, scrubbers, galley power, and power service.
# 
# - Diesel Generators Power Analysis
# 
# Vessel 1:
# 
# The monthly average power generated by Diesel Generators 1-4 shows significant fluctuations throughout the year.
# Diesel Generator 1 saw a gradual decrease until mid-year and then an increase towards the end of the year.
# Diesel Generator 2 had relatively stable power generation with minor fluctuations.
# Diesel Generator 3 showed high variability with peaks during mid-year and at the end.
# Diesel Generator 4 had lower but more consistent power generation.
# 
# Vessel 2:
# 
# Diesel Generators on Vessel 2 also exhibited variability, with Diesel Generator 3 showing the highest fluctuations.
# Diesel Generator 2 displayed a gradual increase over the year.
# Diesel Generators 1 and 4 showed relatively stable trends with minor fluctuations.
# 
# - HVAC Chillers Power Analysis
# 
# Vessel 1:
# 
# The HVAC Chiller power consumption showed notable variability.
# HVAC Chiller 1 had a few peaks during the mid-year but generally maintained lower power consumption.
# HVAC Chiller 2 and 3 had irregular peaks throughout the year, indicating sporadic high usage periods.
# 
# Vessel 2:
# 
# HVAC Chiller 1 showed significant fluctuations, particularly towards the end of the year.
# HVAC Chiller 2 and 3 had erratic usage patterns with notable peaks at various times.
# 
# - Scrubber Power Analysis
# 
# Vessel 1:
# 
# The scrubber power consumption trend decreased during the first half of the year but increased significantly towards the year-end, indicating more extensive use possibly due to stricter emission control periods.
# 
# Vessel 2:
# 
# Scrubber power usage was relatively low but spiked dramatically in the mid and later parts of the year, suggesting intermittent but intense usage periods.
# 
# - Power Galley Analysis
# 
# Vessel 1:
# 
# Power Galley 1 maintained low and stable power consumption.
# Power Galley 2 had higher but consistent power consumption throughout the year.
# 
# Vessel 2:
# 
# Power Galley 1 also showed stable and low power consumption.
# Power Galley 2 experienced higher power usage, with a peak towards the end of the year.
# 
# - Power Service Analysis
# 
# Vessel 1:
# 
# Power service consumption decreased during the first half but increased towards the end of the year, suggesting variations in operational demand or seasonal changes.
# 
# Vessel 2:
# 
# Power service consumption had a steady trend with a significant increase at the end of the year.
# 
# **Conclusion**
# 
# The power generation trends for both vessels indicate variability across different components. Diesel generators, HVAC chillers, and scrubbers show significant fluctuations, likely due to operational demands and environmental compliance requirements. Stable components like power galley indicate consistent energy usage patterns. Monitoring these trends helps in optimizing power management strategies and ensuring efficient vessel operations.

# In[ ]:




