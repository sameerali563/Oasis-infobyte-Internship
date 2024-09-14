#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on retail sales dataset:

# In[33]:


# import libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Load dataset:
data = pd.read_csv('retail_data.csv')
data.head()


# In[4]:


data.info()


# In[5]:


data.tail()


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data.duplicated().sum()


# In[9]:


data.isnull().sum()


# In[10]:


data.astype


# In[11]:


data.drop_duplicates()


# In[12]:


data['Product Category'].value_counts()


# In[13]:


df = pd.DataFrame(data)
df.head()


# In[34]:


df['Date']=pd.to_datetime(df['Date'])


# In[35]:


df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month


# In[36]:


df.head()


# In[37]:


df.describe()


# In[38]:


#Product Category count:
X = df['Product Category'].value_counts()
X


# In[39]:


X.plot(kind='pie',explode=[0.2,0,0.2],figsize=(12,6),autopct='%1.1f%%')
plt.title('Product Catgory')
plt.axis('equal')
plt.show()


# In[40]:


# Product Category Distribution:
sns.countplot(x='Product Category',data=df)
plt.title("Product Category Distribution")
plt.show()


# In[41]:


# Gender count:
gender = df['Gender'].value_counts()
gender


# In[42]:


plt.figure(figsize=(6,6))
plt.pie(gender,autopct='%1.1f%%',explode=(0.1,0))
plt.title('Gender')
plt.axis('equal')
plt.show()


# In[114]:


# DIstribution of Sales:
sns.histplot(df["Total Amount"])
plt.title('DIstribution of Total Sales')


# In[44]:


sns.distplot(df['Total Amount'])
plt.title('Distribution of Sales')


# In[45]:


# Checking Outliers in Total sales;
sns.boxplot(df['Total Amount'])
plt.title("Outliers in sales")


# In[46]:


# Distribution of quantity:
sns.distplot(df['Quantity'])
plt.title('Distribution of Quantity')


# In[120]:


# Distribution of year:
sns.distplot(df['year'])
plt.title('Years wise Trend')


# In[122]:


# Checking outliers in years:
sns.boxplot(df["year"])
plt.show()


# In[49]:


# Monthly DIstribution:
sns.distplot(df["month"])
plt.title('Monthly Trends')


# In[123]:


# Checking outliers in month
sns.boxplot(df['month'])


# In[52]:


#Correlation:
correlation_matrix = df.corr()
correlation_matrix


# In[53]:


# correlation between variables:
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',linewidths='0.5')
plt.title('Correlation between data')
plt.show()


# In[124]:


# Relationship Between Quantity and Total Amount:
plt.figure(figsize=(8,6))
sns.scatterplot(x="Quantity",y="Total Amount",data=df)
plt.title('Sales vs Quantity')
plt.xlabel("Sales")
plt.ylabel("Quantity")
plt.show()


# In[55]:


#pairplot:
sns.pairplot(df)
plt.show()


# In[65]:


#Time Series Analysis over variables:
X= df['month']
Y=df['Total Amount']


# In[66]:


plt.figure(figsize=(10,6))
sns.lineplot(X,Y,data='df',marker='o')
plt.title('Monthly Sales over time')
plt.grid(True)
plt.show()


# In[68]:


#Time Series:
X= df['year']
Y=df['Total Amount']


# In[71]:


plt.figure(figsize=(10,6))
sns.lineplot(X,Y,data='df',marker='o')
plt.title('Yearly Sales over time')
plt.grid(True)
plt.show()


# In[75]:


# Total Sale by each product category:
sale = df.groupby('Product Category')['Total Amount'].sum()


# In[76]:


sale.plot(kind='bar',color='red')
plt.xlabel('Product Category')
plt.ylabel('Total Amount')  
plt.title('Total sales by each Category')
plt.show()


# In[93]:


# Total sale by gender:
gender = df.groupby('Gender')['Total Amount'].sum()


# In[104]:


gender.plot(kind='bar',color='green')
plt.title('Sales by Gender')
plt.show()


# In[81]:


Quantity = df.groupby('Product Category')['Quantity'].mean()


# In[105]:


sns.barplot(data=df, x='Product Category', y='Quantity', palette='viridis')
plt.title('Average Quantity Sales')
plt.show()


# In[107]:


#Total Sales:
Total_sales = df['Total Amount'].sum()
print(f"Total Amount: {Total_sales}")


# In[112]:


# Heighest Selling Product:
product_sales = df.groupby('Product Category')['Total Amount'].sum().reset_index()
highest_selling_product = product_sales.loc[product_sales['Total Amount'].idxmax()]
print(f"Highest-Selling Product: {highest_selling_product['Product Category']} with Total Amount {highest_selling_product['Total Amount']}")

