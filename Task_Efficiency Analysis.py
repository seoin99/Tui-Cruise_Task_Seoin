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

# Plot the time series for 'HVAC Chiller 1 Power (MW)' to see how to handle missing values. 
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


# missing value check 
df.isnull().sum()


# ## Sorting Vessels data to Vessel 1 & Vessel 2

# In[24]:


# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1']
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2']

# Verify the split
print(vessel_1_data.shape)
print(vessel_2_data.shape)


# ## Outlier check of efficiency Analysis

# In[25]:


# Define efficiency columns
efficiency_columns = [
    'Main Engine 1 Fuel Flow Rate (kg/h)', 'Main Engine 2 Fuel Flow Rate (kg/h)',
    'Main Engine 3 Fuel Flow Rate (kg/h)', 'Main Engine 4 Fuel Flow Rate (kg/h)',
    'Boiler 1 Fuel Flow Rate (L/h)', 'Boiler 2 Fuel Flow Rate (L/h)',
    'Incinerator 1 Fuel Flow Rate (L/h)'
]

# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()

# Function to create box plots for outliers visualization
def plot_outliers(df, columns, vessel_name):
    plt.figure(figsize=(20, 10))
    df[columns].boxplot()
    plt.title(f'Outlier Visualization for {vessel_name}', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=0, fontsize=10)  # Change rotation to 0 for horizontal labels
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to make room for the labels
    plt.show()

# Plot outliers for Vessel 1
plot_outliers(vessel_1_data, efficiency_columns, 'Vessel 1')

# Plot outliers for Vessel 2
plot_outliers(vessel_2_data, efficiency_columns, 'Vessel 2')


# ### Each outlier check

# In[26]:


# Define the column of interest for outlier detection
column_of_interest = 'Boiler 1 Fuel Flow Rate (L/h)'

# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()

# Function to identify and display outliers using Z-score method
def identify_outliers(df, column, z_thresh=3):
    z_scores = stats.zscore(df[column])
    abs_z_scores = np.abs(z_scores)
    outliers = df[abs_z_scores > z_thresh]
    return outliers

# Identify outliers in 'Boiler 1 Fuel Flow Rate (L/h)' for Vessel 1
boiler_1_outliers_vessel_1 = identify_outliers(vessel_1_data, column_of_interest)


# In[27]:


# Ensure 'Start Time' is correctly set as the index
vessel_1_data.reset_index(inplace=True)
boiler_1_outliers_vessel_1.reset_index(inplace=True)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(vessel_1_data['Start Time'], vessel_1_data['Boiler 1 Fuel Flow Rate (L/h)'], label='Original Data')
plt.scatter(boiler_1_outliers_vessel_1['Start Time'], boiler_1_outliers_vessel_1['Boiler 1 Fuel Flow Rate (L/h)'], color='red', label='Outliers')
plt.xlabel('Time')
plt.ylabel('Boiler 1 Fuel Flow Rate (L/h)')
plt.title('Boiler 1 Fuel Flow Rate Over Time with Outliers Highlighted')
plt.legend()
plt.grid(True)
plt.show()


# In[28]:


# Ensure 'Start Time' is correctly set as the index
vessel_1_data.set_index('Start Time', inplace=True)
boiler_1_outliers_vessel_1.set_index('Start Time', inplace=True)

# Statistical summary without outliers
data_without_outliers = vessel_1_data[~vessel_1_data.index.isin(boiler_1_outliers_vessel_1.index)]
summary_with_outliers = vessel_1_data['Boiler 1 Fuel Flow Rate (L/h)'].describe()
summary_without_outliers = data_without_outliers['Boiler 1 Fuel Flow Rate (L/h)'].describe()

print("Summary with Outliers:\n", summary_with_outliers)
print("\nSummary without Outliers:\n", summary_without_outliers)


# Analysing with outliers becasue there are not much difference between mean, std, 75% and Max. 

# In[29]:


# Filter data for Vessel 2
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()

# Function to identify outliers
def identify_outliers(df, column):
    z_scores = stats.zscore(df[column])
    abs_z_scores = np.abs(z_scores)
    outliers = df[abs_z_scores > 3]
    return outliers

# Identify outliers in Boiler 1 Fuel Flow Rate for Vessel 2
boiler_1_outliers_vessel_2 = identify_outliers(vessel_2_data, 'Boiler 1 Fuel Flow Rate (L/h)')

# Plot the time series of Boiler 1 Fuel Flow Rate with outliers highlighted for Vessel 2
plt.figure(figsize=(10, 6))
plt.plot(vessel_2_data['Start Time'], vessel_2_data['Boiler 1 Fuel Flow Rate (L/h)'], label='Boiler 1 Fuel Flow Rate (L/h)')
plt.scatter(boiler_1_outliers_vessel_2['Start Time'], boiler_1_outliers_vessel_2['Boiler 1 Fuel Flow Rate (L/h)'], color='red', label='Outliers')
plt.xlabel('Time')
plt.ylabel('Boiler 1 Fuel Flow Rate (L/h)')
plt.title('Boiler 1 Fuel Flow Rate Over Time with Outliers Highlighted (Vessel 2)')
plt.legend()
plt.grid(True)
plt.show()



# In[30]:


# Statistical summary without outliers
data_without_outliers_vessel_2 = vessel_2_data[~vessel_2_data.index.isin(boiler_1_outliers_vessel_2.index)]
summary_with_outliers_vessel_2 = vessel_2_data['Boiler 1 Fuel Flow Rate (L/h)'].describe()
summary_without_outliers_vessel_2 = data_without_outliers_vessel_2['Boiler 1 Fuel Flow Rate (L/h)'].describe()

print("Summary with Outliers for Vessel 2:\n", summary_with_outliers_vessel_2)
print("\nSummary without Outliers for Vessel 2:\n", summary_without_outliers_vessel_2)


# Analysing with outliers becasue outliers consistently shows a distribution

# In[31]:


# Define the column of interest for outlier detection
column_of_interest = 'Incinerator 1 Fuel Flow Rate (L/h)'

# Assuming 'Vessel Name' is the column indicating the vessel
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()

# Function to identify and display outliers using Z-score method
def identify_outliers(df, column, z_thresh=3):
    z_scores = stats.zscore(df[column])
    abs_z_scores = np.abs(z_scores)
    outliers = df[abs_z_scores > z_thresh]
    return outliers

# Identify outliers in 'Incinerator 1 Fuel Flow Rate (L/h)' for Vessel 1
incinerator_1_outliers_vessel_1 = identify_outliers(vessel_1_data, column_of_interest)


# Ensure 'Start Time' is correctly set as the index
vessel_1_data.reset_index(inplace=True)
incinerator_1_outliers_vessel_1.reset_index(inplace=True)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(vessel_1_data['Start Time'], vessel_1_data[column_of_interest], label='Original Data')
plt.scatter(incinerator_1_outliers_vessel_1['Start Time'], incinerator_1_outliers_vessel_1[column_of_interest], color='red', label='Outliers')
plt.xlabel('Time')
plt.ylabel('Incinerator 1 Fuel Flow Rate (L/h)')
plt.title('Incinerator 1 Fuel Flow Rate Over Time with Outliers Highlighted')
plt.legend()
plt.grid(True)
plt.show()


# In[32]:


# Ensure 'Start Time' is correctly set as the index
vessel_1_data.set_index('Start Time', inplace=True)
incinerator_1_outliers_vessel_1.set_index('Start Time', inplace=True)

# Statistical summary without outliers
data_without_outliers = vessel_1_data[~vessel_1_data.index.isin(incinerator_1_outliers_vessel_1.index)]
summary_with_outliers = vessel_1_data[column_of_interest].describe()
summary_without_outliers = data_without_outliers[column_of_interest].describe()

print("Summary with Outliers:\n", summary_with_outliers)
print("\nSummary without Outliers:\n", summary_without_outliers)


# Analysing with outliers becasue outliers consistently shows a distribution

# In[33]:


# Function to plot outliers for 'Main Engine 1 Fuel Flow Rate (kg/h)'
def plot_main_engine_outliers(df, column, vessel_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Start Time'], df[column], label='Original Data')
    plt.scatter(df['Start Time'], df[column], c='blue', label='Data Points')
    outliers = df[df[column] > df[column].mean() + 2 * df[column].std()]
    plt.scatter(outliers['Start Time'], outliers[column], c='red', label='Outliers')
    plt.xlabel('Time')
    plt.ylabel(f'{column}')
    plt.title(f'{column} Over Time with Outliers Highlighted ({vessel_name})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Assuming you have a DataFrame for Vessel 2 similar to vessel_2_data
# Example usage for 'Main Engine 1 Fuel Flow Rate (kg/h)' for 'Vessel 2'
plot_main_engine_outliers(vessel_2_data, 'Main Engine 1 Fuel Flow Rate (kg/h)', 'Vessel 2')




# Analysing with outliers becasue outliers effects really minumum to overall distribution

# # Data Analysis

# In[34]:


# Ensure 'Start Time' is in datetime format and set as index
df['Start Time'] = pd.to_datetime(df['Start Time'])
vessel_1_data = df[df['Vessel Name'] == 'Vessel 1'].copy()
vessel_2_data = df[df['Vessel Name'] == 'Vessel 2'].copy()
vessel_1_data.set_index('Start Time', inplace=True)
vessel_2_data.set_index('Start Time', inplace=True)

# Define efficiency columns
efficiency_columns = [
    'Main Engine 1 Fuel Flow Rate (kg/h)', 'Main Engine 2 Fuel Flow Rate (kg/h)', 
    'Main Engine 3 Fuel Flow Rate (kg/h)', 'Main Engine 4 Fuel Flow Rate (kg/h)', 
    'Boiler 1 Fuel Flow Rate (L/h)', 'Boiler 2 Fuel Flow Rate (L/h)', 
    'Incinerator 1 Fuel Flow Rate (L/h)'
]

# Calculate monthly averages for Vessel 1
vessel_1_monthly_avg = vessel_1_data[efficiency_columns].resample('M').mean()

# Calculate monthly averages for Vessel 2
vessel_2_monthly_avg = vessel_2_data[efficiency_columns].resample('M').mean()

# Plotting monthly averages for Main Engine Fuel Flow Rates for Vessel 1
plt.figure(figsize=(12, 6))
plt.plot(vessel_1_monthly_avg.index, vessel_1_monthly_avg['Main Engine 1 Fuel Flow Rate (kg/h)'], label='Main Engine 1 Fuel Flow Rate (kg/h)')
plt.plot(vessel_1_monthly_avg.index, vessel_1_monthly_avg['Main Engine 2 Fuel Flow Rate (kg/h)'], label='Main Engine 2 Fuel Flow Rate (kg/h)')
plt.plot(vessel_1_monthly_avg.index, vessel_1_monthly_avg['Main Engine 3 Fuel Flow Rate (kg/h)'], label='Main Engine 3 Fuel Flow Rate (kg/h)')
plt.plot(vessel_1_monthly_avg.index, vessel_1_monthly_avg['Main Engine 4 Fuel Flow Rate (kg/h)'], label='Main Engine 4 Fuel Flow Rate (kg/h)')
plt.xlabel('Time')
plt.ylabel('Fuel Flow Rate (kg/h)')
plt.title('Monthly Average Main Engine Fuel Flow Rate Over Time for Vessel 1')
plt.legend()
plt.grid(True)
plt.show()

# Plotting monthly averages for Boiler Fuel Flow Rates for Vessel 1
plt.figure(figsize=(12, 6))
plt.plot(vessel_1_monthly_avg.index, vessel_1_monthly_avg['Boiler 1 Fuel Flow Rate (L/h)'], label='Boiler 1 Fuel Flow Rate (L/h)')
plt.plot(vessel_1_monthly_avg.index, vessel_1_monthly_avg['Boiler 2 Fuel Flow Rate (L/h)'], label='Boiler 2 Fuel Flow Rate (L/h)')
plt.xlabel('Time')
plt.ylabel('Fuel Flow Rate (L/h)')
plt.title('Monthly Average Boiler Fuel Flow Rate Over Time for Vessel 1')
plt.legend()
plt.grid(True)
plt.show()

# Plotting monthly averages for Main Engine Fuel Flow Rates for Vessel 2
plt.figure(figsize=(12, 6))
plt.plot(vessel_2_monthly_avg.index, vessel_2_monthly_avg['Main Engine 1 Fuel Flow Rate (kg/h)'], label='Main Engine 1 Fuel Flow Rate (kg/h)')
plt.plot(vessel_2_monthly_avg.index, vessel_2_monthly_avg['Main Engine 2 Fuel Flow Rate (kg/h)'], label='Main Engine 2 Fuel Flow Rate (kg/h)')
plt.plot(vessel_2_monthly_avg.index, vessel_2_monthly_avg['Main Engine 3 Fuel Flow Rate (kg/h)'], label='Main Engine 3 Fuel Flow Rate (kg/h)')
plt.plot(vessel_2_monthly_avg.index, vessel_2_monthly_avg['Main Engine 4 Fuel Flow Rate (kg/h)'], label='Main Engine 4 Fuel Flow Rate (kg/h)')
plt.xlabel('Time')
plt.ylabel('Fuel Flow Rate (kg/h)')
plt.title('Monthly Average Main Engine Fuel Flow Rate Over Time for Vessel 2')
plt.legend()
plt.grid(True)
plt.show()

# Plotting monthly averages for Boiler Fuel Flow Rates for Vessel 2
plt.figure(figsize=(12, 6))
plt.plot(vessel_2_monthly_avg.index, vessel_2_monthly_avg['Boiler 1 Fuel Flow Rate (L/h)'], label='Boiler 1 Fuel Flow Rate (L/h)')
plt.plot(vessel_2_monthly_avg.index, vessel_2_monthly_avg['Boiler 2 Fuel Flow Rate (L/h)'], label='Boiler 2 Fuel Flow Rate (L/h)')
plt.xlabel('Time')
plt.ylabel('Fuel Flow Rate (L/h)')
plt.title('Monthly Average Boiler Fuel Flow Rate Over Time for Vessel 2')
plt.legend()
plt.grid(True)
plt.show()

# Calculate monthly averages for Incinerator 1 Fuel Flow Rate (L/h) for Vessel 1
vessel_1_monthly_avg_incinerator = vessel_1_data['Incinerator 1 Fuel Flow Rate (L/h)'].resample('M').mean()

# Calculate monthly averages for Incinerator 1 Fuel Flow Rate (L/h) for Vessel 2
vessel_2_monthly_avg_incinerator = vessel_2_data['Incinerator 1 Fuel Flow Rate (L/h)'].resample('M').mean()

# Plotting monthly averages for Incinerator 1 Fuel Flow Rate (L/h) for Vessel 1
plt.figure(figsize=(12, 6))
plt.plot(vessel_1_monthly_avg_incinerator.index, vessel_1_monthly_avg_incinerator, label='Monthly Average Incinerator 1 Fuel Flow Rate (L/h)')
plt.xlabel('Time')
plt.ylabel('Incinerator 1 Fuel Flow Rate (L/h)')
plt.title('Monthly Average Incinerator 1 Fuel Flow Rate Over Time for Vessel 1')
plt.legend()
plt.grid(True)
plt.show()

# Plotting monthly averages for Incinerator 1 Fuel Flow Rate (L/h) for Vessel 2
plt.figure(figsize=(12, 6))
plt.plot(vessel_2_monthly_avg_incinerator.index, vessel_2_monthly_avg_incinerator, label='Monthly Average Incinerator 1 Fuel Flow Rate (L/h)')
plt.xlabel('Time')
plt.ylabel('Incinerator 1 Fuel Flow Rate (L/h)')
plt.title('Monthly Average Incinerator 1 Fuel Flow Rate Over Time for Vessel 2')
plt.legend()
plt.grid(True)
plt.show()


# In[35]:


# Calculate the correlation matrix
correlation_matrix = df[efficiency_columns].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Fuel Flow Rates')
plt.show()


# In[36]:


# Plotting and storing analysis results as DataFrames
results = []

# Efficiency Analysis for Vessel 1
efficiency_vessel_1 = vessel_1_monthly_avg.copy()
efficiency_vessel_1['Vessel'] = 'Vessel 1'
results.append(efficiency_vessel_1)

# Efficiency Analysis for Vessel 2
efficiency_vessel_2 = vessel_2_monthly_avg.copy()
efficiency_vessel_2['Vessel'] = 'Vessel 2'
results.append(efficiency_vessel_2)

# Combine all results
combined_results = pd.concat(results)

# Store combined results into SQLite database
combined_results.to_sql('efficiency_analysis', conn, if_exists='replace', index=True)

# Verify data is stored correctly
query = "SELECT * FROM efficiency_analysis"
df_sql = pd.read_sql_query(query, conn)
print(df_sql.head())

# Close the connection
conn.close()


# # Efficiency Performance Trend Analysis 
# 
# 
# **Introduction**
# This report provides an in-depth analysis of the efficiency performance trends for two cruise ships, Vessel 1 and Vessel 2, based on their fuel flow rates for various engines and boilers. The analysis covers the period from early 2023 to early 2024, focusing on monthly average fuel flow rates for main engines, boilers, and incinerators.
# 
# **Data Preparation**
# 
# The dataset includes the following columns:
# 
# Main Engine 1 Fuel Flow Rate (kg/h)
# Main Engine 2 Fuel Flow Rate (kg/h)
# Main Engine 3 Fuel Flow Rate (kg/h)
# Main Engine 4 Fuel Flow Rate (kg/h)
# Boiler 1 Fuel Flow Rate (L/h)
# Boiler 2 Fuel Flow Rate (L/h)
# Incinerator 1 Fuel Flow Rate (L/h)
# 
# The dataset was processed to handle missing values using linear interpolation and ensure all necessary columns were in numeric format.
# 
# **Analysis and Findings**
# 
# **Main Engine Fuel Flow Rates for Vessel 1:**
# 
# - Monthly Average Trends:
# There is a noticeable decline in the fuel flow rates for Main Engine 1 and Main Engine 2 from January to July 2023.
# Main Engine 3 and Main Engine 4 show more variability with spikes in May, July, and December 2023.
# The variability suggests possible changes in operational patterns or maintenance activities during these periods.
# 
# - Interpretation:
# The decline in fuel flow rates for Main Engine 1 and Main Engine 2 may indicate improvements in operational efficiency or reduced load on these engines.
# The spikes in Main Engine 3 and Main Engine 4 might be due to increased demand or periods of higher operational intensity.
# 
# **Boiler Fuel Flow Rates for Vessel 1:**
# 
# - Monthly Average Trends:
# Both Boiler 1 and Boiler 2 show significant fluctuations throughout the year.
# Boiler 1 experienced peaks in April and August 2023, while Boiler 2 had peaks in August and December 2023.
# 
# - Interpretation:
# The peaks in fuel flow rates for the boilers may correspond to periods of increased onboard heating or hot water demand.
# The fluctuations suggest varying operational requirements and possibly seasonal changes impacting boiler usage.
# 
# **Main Engine Fuel Flow Rates for Vessel 2:**
# 
# - Monthly Average Trends:
# Main Engine 1 and Main Engine 2 fuel flow rates show an increasing trend from May to October 2023, followed by a decline towards the end of the year.
# Main Engine 3 and Main Engine 4 also exhibit similar variability with peaks in mid-2023.
# 
# - Interpretation:
# The increasing trend in mid-2023 indicates periods of higher operational activity or less efficient operation during these months.
# The decline at the end of the year could be attributed to reduced operational demand or efficiency improvements.
# 
# **Boiler Fuel Flow Rates for Vessel 2:**
# 
# - Monthly Average Trends:
# Boiler 1 fuel flow rate peaked in March 2023 and then showed a declining trend.
# Boiler 2 exhibited peaks in March and November 2023, with a general decline in other months.
# 
# - Interpretation:
# The peaks in Boiler 1 and Boiler 2 suggest periods of high demand for heating or other boiler-related services.
# The overall decline could be due to improved efficiency or reduced operational needs.
# 
# **Incinerator Fuel Flow Rates:**
# 
# - Vessel 1:
# The fuel flow rate for Incinerator 1 shows a declining trend from January to June 2023, followed by fluctuations for the rest of the year.
# 
# - Vessel 2:
# The fuel flow rate for Incinerator 1 fluctuates throughout the year, with peaks in mid-2023.
# Interpretation:
# 
# The declining trend for Vessel 1 suggests reduced incinerator usage or improved waste management practices.
# Fluctuations in Vessel 2 may indicate variable waste generation rates or differing operational needs.
# 
# **Correlation Analysis of Fuel Flow Rates:**
# 
# The correlation matrix shows the relationships between the fuel flow rates of different engines and boilers.
# 
# **Key Insights:**
# 
# There is a negative correlation between Main Engine 1 and Main Engine 2 fuel flow rates, indicating that when one engine's fuel flow rate increases, the other's tends to decrease.
# Main Engine 3 and Main Engine 4 have a moderate negative correlation with Boiler 1 and Boiler 2 fuel flow rates, suggesting that higher fuel usage in the main engines might be associated with lower fuel usage in the boilers.
# Incinerator 1 fuel flow rate shows weak correlations with other components, indicating independent operation.
# 
# **Conclusion**
# 
# Both vessels exhibit variability in fuel flow rates across different engines and boilers, indicating changes in operational patterns, demand, and possibly maintenance activities.
# Peaks in fuel flow rates often correspond to periods of higher operational activity or increased demand for specific services (e.g., heating).
# Declining trends in some fuel flow rates suggest improvements in efficiency or reduced operational demand over time.
# The correlation analysis highlights relationships between the fuel usage of different components, providing insights into operational dependencies and efficiencies.
