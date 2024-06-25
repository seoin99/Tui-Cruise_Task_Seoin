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
##interpolation is a process of determining the unknown values that lie in between the known data points
df['HVAC Chiller 1 Power (MW)'] = df['HVAC Chiller 1 Power (MW)'].interpolate()
df['HVAC Chiller 2 Power (MW)'] = df['HVAC Chiller 2 Power (MW)'].interpolate()
df['HVAC Chiller 3 Power (MW)'] = df['HVAC Chiller 3 Power (MW)'].interpolate()

# If you need to reset the index back to columns
df.reset_index(inplace=True)


# In[16]:


df[df['Power Service (MW)'].isna()]


# In[17]:


# First forward fill, then backward fill
# # the next available value after the missing data point replaces the missing value because it is only two missing values
df['Power Service (MW)'] = df['Power Service (MW)'].ffill().bfill()



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


# In[22]:


# Set 'Start Time' as the index
df.set_index('Start Time', inplace=True)

# Interpolate to fill missing values
df['Speed Through Water (knots)'] = df['Speed Through Water (knots)'].interpolate()

# If you need to reset the index back to columns
df.reset_index(inplace=True)


# In[23]:


# # missing value check 
df.isnull().sum()


# ## Sorting Vessels data to Vessel 1 & Vessel 2

# In[24]:


# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1']
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2']

# Verify the split
print(vessel_1_data.shape)
print(vessel_2_data.shape)


# In[25]:


df.columns


# ## Outlier check of Propulsion  Analysis  

# In[26]:


# Define efficiency columns
propulsion_columns = [
     'Propulsion Power (MW)', 'Port Side Propulsion Power (MW)', 
    'Starboard Side Propulsion Power (MW)', 'Bow Thruster 1 Power (MW)', 
    'Bow Thruster 2 Power (MW)', 'Bow Thruster 3 Power (MW)', 
    'Stern Thruster 1 Power (MW)', 'Stern Thruster 2 Power (MW)', 
    'Speed Over Ground (knots)', 'Speed Through Water (knots)'
]

# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()

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
plot_outliers(vessel_1_data, propulsion_columns, 'Vessel 1')

# Plot outliers for Vessel 2
plot_outliers(vessel_2_data, propulsion_columns, 'Vessel 2')


# ### Each outlier check

# In[27]:


import numpy as np
import matplotlib.pyplot as plt

# Ensure 'Start Time' is in datetime format and set as index
df['Start Time'] = pd.to_datetime(df['Start Time'])
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()
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

# Detect outliers, plot, and print summaries for each relevant column in vessel_1_data
columns_to_check = ['Bow Thruster 2 Power (MW)', 'Bow Thruster 3 Power (MW)', 
    'Stern Thruster 1 Power (MW)', 'Stern Thruster 2 Power (MW)']

for column in columns_to_check:
    outliers = detect_outliers(vessel_1_data, column)
    plot_with_outliers(vessel_1_data, column, outliers, 'Vessel 1')
    print_summaries(vessel_1_data, column, outliers)


# Analysing with outliers becasue outliers consistently shows a distribution

# # Data Analysis

# In[28]:


# Define categories
total_propulsion = ['Propulsion Power (MW)']
side_propulsion = ['Port Side Propulsion Power (MW)', 'Starboard Side Propulsion Power (MW)']
bow_thrusters = ['Bow Thruster 1 Power (MW)', 'Bow Thruster 2 Power (MW)', 'Bow Thruster 3 Power (MW)']
stern_thrusters = ['Stern Thruster 1 Power (MW)', 'Stern Thruster 2 Power (MW)']
speed_metrics = ['Speed Over Ground (knots)', 'Speed Through Water (knots)']

# Combine all columns into a single list
propulsion_columns = total_propulsion + side_propulsion + bow_thrusters + stern_thrusters + speed_metrics

# Ensure all columns are numeric
df[propulsion_columns] = df[propulsion_columns].apply(pd.to_numeric, errors='coerce')

# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()

# Calculate monthly averages for Vessel 1
vessel_1_data.set_index('Start Time', inplace=True)
vessel_1_monthly_avg = vessel_1_data[propulsion_columns].resample('M').mean()

# Calculate monthly averages for Vessel 2
vessel_2_data.set_index('Start Time', inplace=True)
vessel_2_monthly_avg = vessel_2_data[propulsion_columns].resample('M').mean()

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
plot_monthly_averages_vessel_1(total_propulsion, 'Total Propulsion Power')
plot_monthly_averages_vessel_1(side_propulsion, 'Side Propulsion Power')
plot_monthly_averages_vessel_1(bow_thrusters, 'Bow Thruster Power')
plot_monthly_averages_vessel_1(stern_thrusters, 'Stern Thruster Power')
plot_monthly_averages_vessel_1(speed_metrics, 'Speed Metrics')

# Plotting for each category for Vessel 2
plot_monthly_averages_vessel_2(total_propulsion, 'Total Propulsion Power')
plot_monthly_averages_vessel_2(side_propulsion, 'Side Propulsion Power')
plot_monthly_averages_vessel_2(bow_thrusters, 'Bow Thruster Power')
plot_monthly_averages_vessel_2(stern_thrusters, 'Stern Thruster Power')
plot_monthly_averages_vessel_2(speed_metrics, 'Speed Metrics')


# In[29]:


# Define the columns for correlation analysis
propulsion_columns = [
    'Propulsion Power (MW)', 'Port Side Propulsion Power (MW)', 
    'Starboard Side Propulsion Power (MW)', 
    'Bow Thruster 2 Power (MW)', 'Bow Thruster 3 Power (MW)', 
    'Stern Thruster 1 Power (MW)', 'Stern Thruster 2 Power (MW)', 
    'Speed Over Ground (knots)', 'Speed Through Water (knots)'
]

# Calculate the correlation matrix
correlation_matrix = df[propulsion_columns].corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Propulsion Power and Speed Metrics')
plt.show()


# In[32]:


# Define categories for propulsion columns
total_propulsion = ['Propulsion Power (MW)']
side_propulsion = ['Port Side Propulsion Power (MW)', 'Starboard Side Propulsion Power (MW)']
bow_thrusters = ['Bow Thruster 1 Power (MW)', 'Bow Thruster 2 Power (MW)', 'Bow Thruster 3 Power (MW)']
stern_thrusters = ['Stern Thruster 1 Power (MW)', 'Stern Thruster 2 Power (MW)']
speed_metrics = ['Speed Over Ground (knots)', 'Speed Through Water (knots)']

# Combine all columns into a single list for propulsion analysis
propulsion_columns = total_propulsion + side_propulsion + bow_thrusters + stern_thrusters

# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()

# Convert index to datetime if it's not already
vessel_1_data.index = pd.to_datetime(vessel_1_data.index)
vessel_2_data.index = pd.to_datetime(vessel_2_data.index)

# Calculate monthly averages for Vessel 1
vessel_1_monthly_avg = vessel_1_data[propulsion_columns].resample('M').mean()

# Calculate monthly averages for Vessel 2
vessel_2_monthly_avg = vessel_2_data[propulsion_columns].resample('M').mean()

# Add 'Vessel' column to identify each vessel
vessel_1_monthly_avg['Vessel'] = 'Vessel 1'
vessel_2_monthly_avg['Vessel'] = 'Vessel 2'

# Combine all results
combined_results = pd.concat([vessel_1_monthly_avg, vessel_2_monthly_avg])

# Store combined results into SQLite database
conn = sqlite3.connect('your_database.db')
combined_results.to_sql('monthly_propulsion_data', conn, if_exists='replace', index=True)

# Verify data is stored correctly
query = "SELECT * FROM monthly_propulsion_data"
df_sql = pd.read_sql_query(query, conn)
print(df_sql.head())

# Close the connection
conn.close()


# # Performance Analysis Report: Propulsion Power and Speed Metrics
# 
# **Introduction**
# This report presents an analysis of propulsion power and speed metrics for two vessels over a defined period. The aim is to understand the performance trends, correlations among various power and speed-related parameters, and insights for operational efficiency.
# 
# **Data Preparation**
# 
# Data was filtered for two specific vessels.
# Missing values were handled through interpolation.
# 
# Weekly averages were computed for each metric.
# 
# The dataset includes the following columns:
# 
# - Total Propulsion Power:
# Propulsion Power (MW)
# 
# - Side Propulsion Power:
# Port Side Propulsion Power (MW)
# Starboard Side Propulsion Power (MW)
# 
# - Thruster Power:
# Bow Thruster 1 Power (MW)
# Bow Thruster 2 Power (MW)
# Bow Thruster 3 Power (MW)
# Stern Thruster 1 Power (MW)
# Stern Thruster 2 Power (MW)
# 
# - Speed Metrics:
# Speed Over Ground (knots)
# Speed Through Water (knots)
# 
# **Insights for Performance Analysis:**
# 
# Monitoring Propulsion Power and Speed:
# 
# Given the strong correlations, monitoring the total propulsion power will give a good indication of the vessel's speed. This is crucial for assessing the efficiency and performance of the vessel.
# 
# Thruster Operations:
# 
# Since the thrusters show weak correlations with propulsion power and speed, their use might be more operation-specific, such as docking, undocking, and maneuvering in constrained spaces. This could be further investigated by correlating thruster usage with specific operational events or locations.
# 
# **Propulsion Power Performance Trend Analysis**
# 
# This section provides a detailed analysis of the propulsion power performance trends for Vessel 1 and Vessel 2. The analysis covers the following categories:
# 
# - Total Propulsion Power
# 
# Vessel 1: The total propulsion power shows significant fluctuations over the year, with peaks observed in January, March, and towards the end of the year.
# Vessel 2: The total propulsion power also fluctuates but shows a more pronounced peak during the summer months (June to August) and a significant peak in November.
# 
# - Side Propulsion Power (Port Side and Starboard Side)
# 
# Vessel 1: Both port side and starboard side propulsion powers follow a closely synchronized pattern, with peaks and troughs aligned.
# Vessel 2: Similar synchronization is observed, with increased usage around mid-year and a sharp peak in November.
# 
# - Bow Thrusters Power (Bow Thruster 1, 2, 3)
# 
# Vessel 1: Bow thruster power usage shows frequent peaks, particularly around March and mid-year.
# Vessel 2: Bow thruster power usage shows more pronounced fluctuations with higher peaks mid-year and towards the end of the year.
# 
# - Stern Thrusters Power (Stern Thruster 1, 2)
# 
# Vessel 1: Stern thruster power usage shows noticeable peaks, especially in the first and third quarters, with synchronized usage between the two thrusters.
# Vessel 2: Similar synchronized usage patterns, with notable decreases in usage towards the end of the year.
# 
# - Speed Metrics (Speed Over Ground and Speed Through Water)
# 
# Vessel 1: Speed over ground and speed through water follow closely similar trends, with peaks at the beginning and mid-year.
# Vessel 2: Similar patterns, with synchronized speeds and a sharp peak in November.
# 
# **Summary**
# 
# - Seasonal Influence: Both vessels exhibit seasonal trends in power usage and speeds, with higher values generally observed mid-year and specific peaks around November and December.
# 
# - Operational Patterns: Synchronized usage of propulsion systems (port, starboard, bow, and stern thrusters) suggests coordinated maneuvers or operational activities.
# 
# - Efficiency: Vessel 2 shows a more pronounced peak in total propulsion power mid-year, whereas Vessel 1 shows more fluctuations throughout the year. This could indicate different operational strategies or conditions faced by the vessels.
# 
# 
# **Correlation Analysis**
# 
# Correlation with Speed Metrics:
# 
# There is a very strong positive correlation between Propulsion Power (MW), Port Side Propulsion Power (MW), Starboard Side Propulsion Power (MW), and both speed metrics (Speed Over Ground (knots) and Speed Through Water (knots)), with coefficients around 0.90 to 0.91. 
# This indicates that as the propulsion power increases, the speed of the vessel, both over ground and through water, also increases significantly.
# 
# Weak Correlation with Thrusters:
# 
# Bow Thruster 1 Power (MW) has missing values (denoted by -) indicating that there might be no data available for this metric, or it could be a constant value across the dataset.
# 
# Bow Thruster 2 Power (MW), Bow Thruster 3 Power (MW), Stern Thruster 1 Power (MW), and Stern Thruster 2 Power (MW) show weak negative correlations with the propulsion power metrics and speed metrics (values ranging from -0.09 to -0.15). This suggests that the thrusters' power does not significantly increase with the propulsion power or speed, which makes sense as thrusters are typically used for maneuvering rather than continuous propulsion.
# 
# **Conclusion**
# 
# The analysis of propulsion power performance trends reveals that both Vessel 1 and Vessel 2 exhibit distinct seasonal and operational patterns in their power usage and speeds. Understanding these trends can aid in optimizing operational efficiency, planning maintenance, and developing strategies to manage power usage effectively throughout the year.
# 

# In[ ]:




