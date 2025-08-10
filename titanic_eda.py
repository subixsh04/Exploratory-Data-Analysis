import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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

#data of people who survived based on their gender
print(data.groupby('Sex')['Survived'].mean())

#graph data showing people who survived based on their gender
sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.show()

#survival rate of passengers by the class they were seated in
print(data.groupby('PassengerClass')['Survived'].mean())

#graphical data of the survival rate of passengers by the class they were seated in
sns.barplot(x='PassengerClass', y='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.show()

#survival rates of each gender-class combination
print(data.groupby(['PassengerClass', 'Sex'])['Survived'].mean())

#graphical data of survival rates of gender-class combination 
sns.barplot(x='PassengerClass', y='Survived', hue='Sex', data=data)
plt.title('Survival Rate by Passenger Class and Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.show()

#describes the mean, median, min, max etc quantities
print(data['Age'].describe())

#using a histogram to understand the survival rate of passengers across age
sns.histplot(data['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

#comparison of survivors in a plot across different age using boxplot
sns.boxplot(x='Survived', y='Age', data=data)
plt.title('Age vs Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()

#Histogram that indicates how the price of the ticket has influenced the survival of the passengers
sns.histplot(data['Fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.show()

#boxplot for the price vs survival
sns.boxplot(x='Survived', y='Fare', data=data)
plt.title('Fare vs Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Fare')
plt.show()

#now we are moving onto training the algorithm according to our goal
features = ['PassengerClass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = data[features].copy()
y = data['Survived'] 

#Label encoding for certain categories as ML doesn't process and train string data
label_encoder = LabelEncoder()
X['Sex'] = label_encoder.fit_transform(X['Sex'])

#one-hot encoding the embarked category
X = pd.get_dummies(X, columns=['Embarked'])

#scaling the numerical cloumns
scaler = StandardScaler()
num_cols = ['PassengerClass', 'Age', 'SibSp', 'Parch', 'Fare']
X[num_cols] = scaler.fit_transform(X[num_cols])

#checking the first couple rows after the scaling
print(X.head())

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 20% for testing
    random_state=42,      # reproducible split
    stratify=y            # preserve class proportions
)

print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test  shape:", y_test.shape)

# are class proportions similar?
print("\nTrain class balance (%):")
print((y_train.value_counts(normalize=True) * 100).round(2))
print("\nTest class balance (%):")
print((y_test.value_counts(normalize=True) * 100).round(2))


