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


# # Step : Data Understanding

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


# Drop rows where 'Power Galley 1 (MW)' has missing values
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
df['HVAC Chiller 1 Power (MW)'] = df['HVAC Chiller 1 Power (MW)'].interpolate()
df['HVAC Chiller 2 Power (MW)'] = df['HVAC Chiller 2 Power (MW)'].interpolate()
df['HVAC Chiller 3 Power (MW)'] = df['HVAC Chiller 3 Power (MW)'].interpolate()

# If you need to reset the index back to columns
df.reset_index(inplace=True)


# In[16]:


df[df['Power Service (MW)'].isna()]


# In[17]:


# First forward fill, then backward fill
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


# ## Outlier check and Analysis

# In[26]:


# Define columns to convert to numeric
columns_to_convert = ['Boiler 1 Fuel Flow Rate (L/h)', 'Boiler 2 Fuel Flow Rate (L/h)', 
                      'Main Engine 1 Fuel Flow Rate (kg/h)', 'Main Engine 2 Fuel Flow Rate (kg/h)', 
                      'Main Engine 3 Fuel Flow Rate (kg/h)', 'Main Engine 4 Fuel Flow Rate (kg/h)', 
                      'Incinerator 1 Fuel Flow Rate (L/h)', 'Sea Temperature (Celsius)',
                      'Relative Wind Angle (Degrees)', 'True Wind Angle (Degrees)', 
                      'Relative Wind Direction (Degrees)', 'True Wind Direction (Degrees)', 
                      'True Wind Speed (knots)', 'Relative Wind Speed (knots)']

# Ensure all necessary columns are numeric
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Set 'Start Time' as datetime index
df['Start Time'] = pd.to_datetime(df['Start Time'], errors='coerce')
df.set_index('Start Time', inplace=True)

# Interpolate missing values linearly
df[columns_to_convert] = df[columns_to_convert].interpolate(method='linear')

# Define emission factors
emission_factors = {
    'main_engine': 3.17,  # kg CO2 per kg of fuel
    'boiler': 2.68,       # kg CO2 per liter of fuel
    'incinerator': 2.68   # kg CO2 per liter of fuel
}

# Calculate emissions
df['Main Engine 1 CO2 (kg)'] = df['Main Engine 1 Fuel Flow Rate (kg/h)'] * emission_factors['main_engine']
df['Main Engine 2 CO2 (kg)'] = df['Main Engine 2 Fuel Flow Rate (kg/h)'] * emission_factors['main_engine']
df['Main Engine 3 CO2 (kg)'] = df['Main Engine 3 Fuel Flow Rate (kg/h)'] * emission_factors['main_engine']
df['Main Engine 4 CO2 (kg)'] = df['Main Engine 4 Fuel Flow Rate (kg/h)'] * emission_factors['main_engine']
df['Boiler 1 CO2 (kg)'] = df['Boiler 1 Fuel Flow Rate (L/h)'] * emission_factors['boiler']
df['Boiler 2 CO2 (kg)'] = df['Boiler 2 Fuel Flow Rate (L/h)'] * emission_factors['boiler']
df['Incinerator 1 CO2 (kg)'] = df['Incinerator 1 Fuel Flow Rate (L/h)'] * emission_factors['incinerator']

# Sum emissions to get total emissions
df['Total CO2 Emissions (kg)'] = (df['Main Engine 1 CO2 (kg)'] +
                                   df['Main Engine 2 CO2 (kg)'] +
                                   df['Main Engine 3 CO2 (kg)'] +
                                   df['Main Engine 4 CO2 (kg)'] +
                                   df['Boiler 1 CO2 (kg)'] +
                                   df['Boiler 2 CO2 (kg)'] +
                                   df['Incinerator 1 CO2 (kg)'])

# Filter dataframe to include only numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate monthly averages
monthly_avg = df[numeric_columns].resample('M').mean()


# Correlation Analysis
correlation_matrix = monthly_avg[columns_to_convert + ['Total CO2 Emissions (kg)', 'Sea Temperature (Celsius)']].corr()


# Plotting correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Weather Conditions, Fuel Flow Rates, Sea Temperature, and CO2 Emissions')
plt.show()



# # Environmental Impact Analysis Report
# 
# **Introduction**
# 
# This report analyzes the environmental impact of two cruise vessels by examining the correlations between fuel flow rates, emissions, sea temperature, and weather conditions. Using fuel flow rates as proxies for emissions, the study evaluates how different operational and environmental factors influence CO2 emissions.
# 
# **Data and Methods**
# 
# The analysis is based on the following key variables:
# 
# - Fuel Flow Rates:
# 
# Boiler 1 Fuel Flow Rate (L/h)
# Boiler 2 Fuel Flow Rate (L/h)
# Main Engine Fuel Flow Rates (1-4) (kg/h)
# Incinerator 1 Fuel Flow Rate (L/h)
# 
# - Environmental Conditions:
# 
# Sea Temperature (Celsius)
# Relative Wind Angle (Degrees)
# True Wind Angle (Degrees)
# Relative Wind Direction (Degrees)
# True Wind Direction (Degrees)
# True Wind Speed (knots)
# Relative Wind Speed (knots)
# 
# - Emissions:
# 
# Total CO2 Emissions (kg)
# The emissions data were derived using the fuel flow rates and respective 
# 
# - emission factors:
# 
# Main Engine: 3.17 kg CO2 per kg of fuel
# Boiler: 2.68 kg CO2 per liter of fuel
# Incinerator: 2.68 kg CO2 per liter of fuel
# 
# **Results**
# 
# - Correlation Analysis:
# 
# Total CO2 Emissions show a strong positive correlation with the fuel flow rates of the main engines, especially with Main Engine 4 (0.85), Main Engine 3 (0.53), Main Engine 2 (0.44), and Main Engine 1 (0.41). This indicates that the main engines are significant contributors to CO2 emissions.
# 
# Sea Temperature has a moderate positive correlation with wind speeds, such as True Wind Speed (0.68) and Relative Wind Speed (0.53), suggesting that higher sea temperatures are associated with stronger winds.
# 
# There is a weak positive correlation between Sea Temperature and Total CO2 Emissions (0.12), indicating a slight increase in CO2 emissions with rising sea temperatures.
# 
# - Weather Conditions:
# 
# Wind angles and directions show high correlations among themselves, indicating consistent measurements.
# 
# Wind speeds show moderate correlations with sea temperature, suggesting environmental factors play a role in the operational efficiency and fuel consumption of the vessels.
# 
# - Trends Over Time
# 
# Monthly Averages:
# 
# CO2 emissions and sea temperature data were plotted over time to observe seasonal trends.
# Fuel flow rates for the main engines and boilers generally show consistent usage patterns with some peaks, likely corresponding to periods of higher operational demand or specific voyages.
# Sea temperatures show a clear seasonal trend, increasing during warmer months and decreasing during colder months.
# True and relative wind speeds also exhibit variability over time, with higher speeds often associated with higher sea temperatures.
# 
# **Conclusion**
# 
# The environmental impact analysis of the two cruise vessels underscores the significant role of main engines in CO2 emissions and the influence of sea temperature and weather conditions on operational efficiency. Continuous monitoring and adaptive strategies are essential for reducing emissions and enhancing environmental performance.
# 

# In[ ]:




