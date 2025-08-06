import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Reading the csv file
data = pd.read_csv('train.csv')

#print the first few rows
print(data.head())

#printing the data types/ other non-value information from the csv file
print(data.info())

#checking how many missing data is there in the file using pandas NaN function and returning the number of total missing values
print(data.isnull().sum())

#calculation the missing percentage of the total file
missing_percent = (data.isnull().sum() / len(data)) * 100
print(missing_percent)

#we have now decided from the dataset that the cabin column has too many missing values so we're dropping it
#and we also saw that there is only ~20% of Age is missing, but it is an important column so we are going to fill them using median or mean method
#then, we also noticed that the embarked column has less than 1% of data missing hence we can fill them up using mode method

#dropping the cabin column
data = data.drop(columns=['Cabin'])

#filling in the missing Age data using median method
data['Age'] = data['Age'].fillna(data['Age'].median())

#filling in the missing data of Embarked column
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

#seeing the change in data after the dropping of cloumns and filling them up
print(data.isnull().sum())

#moving on the reforming, dropping and refformatting the data
#we're dropping the passengerId and ticket column as they are not in any form relevant to our anyalsis.
data = data.drop(columns=['PassengerId', 'Ticket'])

#renaming passenger class
data = data.rename(columns={'Pclass': 'PassengerClass'})

#printing the updated version of column list
print(data.columns)

#counting the frequency of the survival rate and printing the percentage of them
print(data['Survived'].value_counts()) #count value
print(data['Survived'].value_counts(normalize=True) * 100) # percentage value

#now we are showing this survival rate data as a graph to visualize
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.show()

#now we are displaying the survival rates as a pie chart to show their percentage
# Calculate survival percentages
survival_counts = data['Survived'].value_counts()
labels = ['Did Not Survive', 'Survived']
colors = ['lightcoral', 'skyblue']  # optional: makes it look nice

# Creating a pie chart
plt.figure(figsize=(6,6))
plt.pie(
    survival_counts,
    labels=labels,
    autopct='%1.1f%%',  # showing percentages of each compartment
    startangle=90,
    colors=['lightcoral', 'skyblue'],
    shadow=True,
    explode=[0, 0.1] 
)
plt.title('Survival Distribution')
plt.axis('equal')  # Equal aspect ratio ensures circle shape
plt.show()


