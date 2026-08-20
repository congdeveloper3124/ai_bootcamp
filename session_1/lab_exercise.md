# Lab Exercise: Your First ML Model

In this exercise, you'll train a simple classifier using the classic Iris dataset.

## Steps

1. **Import the required libraries**

   ```python
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.metrics import accuracy_score
   ```

2. **Load the dataset**

   ```python
   data = load_iris()
   X, y = data.data, data.target
   ```

3. **Split into train/test sets**

   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

4. **Train a Decision Tree model**

   ```python
   model = DecisionTreeClassifier()
   model.fit(X_train, y_train)
   ```

5. **Evaluate the model**

   ```python
   predictions = model.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, predictions))
   ```

## ✅ Success Criteria

- Model runs without errors
- Accuracy score is printed and is above 90%

## 💡 Bonus Challenge

Try swapping the `DecisionTreeClassifier` for a `RandomForestClassifier` or `KNeighborsClassifier` and compare accuracy scores.
