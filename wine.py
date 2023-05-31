import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from PIL import Image


wine = pd.read_csv("C:/Users/Dell/OneDrive/Desktop/code clause/WineQT.csv")


st.set_page_config(layout='wide',initial_sidebar_state='collapsed')
st.header("WINE_QUALITY_PREDICTIONS APP")

menu = [ 'Home🏡','Dataset📂','Model Performance🔍', 'Predict Wine Quality🍷' ,'About Me👨‍💻']

choice = st.radio('Select an option', menu)

if choice == "Home🏡":
       st.markdown('The wine quality data predictions refer to the task of using machine learning algorithms to predict the quality of a wine based on its various features or variables. This dataset typically contains information about the chemical properties of the wine, such as acidity, alcohol content, and residual sugar, as well as a subjective rating of the wines quality, usually on a scale from 0 to 10')
       image = Image.open(r"C:/Users/Dell/OneDrive/Desktop/code clause/wine 2.jpg")
       st.image(image)


if choice == "About Me👨‍💻":
      st.header("Project Designed by Jagadish_Vemula ")
      st.markdown("""
              ## Data Science Enthusiast
            - 📫 You can reach me on my email: jagadeesh12337@gmail.com""")
      st.write('Contact Information')
      st.write('📭jagadeesh12337@gmail.com')
      st.write('Connect with me on:')
      st.write('Click On the Website: ')
      st.write( 'Linkdin_id :' 'https://www.linkedin.com/in/jagadish-vemula-599415206/')
      st.write('My WorkSpace')
      st.write( 'Github_id  :'  'https://github.com/vemula2594')

if choice == "Dataset📂":
 st.write(wine)
 st.markdown('fixed acidity refers to the amount of non-volatile acids in the wine')
 st.markdown('volatile acidity is a measure of the amount of acetic acid in the wine, which contributes to the sour taste')
 st.markdown('citric acid is a weak organic acid that is often added to wine for flavor or acidity')
 st.markdown('residual sugar refers to the amount of sugar remaining in the wine after fermentation is complete')
 st.markdown('chlorides are the salts of hydrochloric acid, which can contribute to the taste of the wine')
 st.markdown('free sulfur dioxide and total sulfur dioxide are measures of the amount of sulfur dioxide in the wine, which acts as a preservative and antioxidant')
 st.markdown('density is a measure of the mass per unit volume of the wine')
 st.markdown('pH is a measure of the acidity or basicity of the wine')
 st.markdown('sulphates are compounds that can be added to wine as a preservative or stabilizer')
 st.markdown('alcohol refers to the percentage of alcohol in the wine.')
 st.markdown('quality refers to a subjective rating of the wines quality')
 st.markdown('Id is likely a unique identifier assigned to each observation in the dataset')  


if choice == 'Model Performance🔍':
    X = wine.drop(['quality','Id'] , axis = 1)
    y = wine['quality']

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    rf_pred_train = rf.predict(X_train)
    rf_pred_test = rf.predict(X_test)
    col1 , col2 , col3 , col4 = st.columns(4)
    with col1:
        st.write(rf_pred_train)
        st.caption("Training Predictions")
    with col2:
        st.write(rf_pred_test)
        st.caption("Test Predictions")

    with col3:
        from sklearn.metrics import f1_score , classification_report , accuracy_score , precision_score , recall_score
        st.subheader("Accuracy Score")
        st.write(accuracy_score(y_test, rf_pred_test))
        st.subheader("Precision Score")
        st.write(precision_score(y_test, rf_pred_test, average='weighted'))
        st.subheader("Recall Score")
        st.write(recall_score(y_test, rf_pred_test, average='weighted'))
        st.subheader("F1 Score")
        st.write(f1_score(y_test, rf_pred_test, average='weighted'))

    with col4:
        score = rf.score(X_train, y_train)
        st.write( "Random Forest Train Accuracy: {:.2f}%".format(score*100))
        score = rf.score(X_test, y_test)
        st.write(" Random Forest Test Accuracy: {:.2f}%".format(score*100))

        from sklearn.metrics import plot_confusion_matrix
        import warnings
        warnings.filterwarnings('ignore')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        cf_rf = plot_confusion_matrix(rf, X_test, y_test)
        st.pyplot()
        st.caption('Confusion matrix for Random Forest classifier')


    st.subheader("Random Forest Model")
    # create a slider to select the tree index
    tree_index = st.slider("Select a tree index:", 0, 99, 4)
    # plot the selected tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(rf.estimators_[tree_index], filled=True, ax=ax)
    st.pyplot(fig)


if choice == 'Predict Wine Quality🍷':
    st.title("Predicting Wine Quality")
    col1 , col2 = st.columns(2)
    st.write("Please enter the following information about the wine:")
    with col1:
           fixed_acidity = st.number_input("Fixed Acidity")
           volatile_acidity = st.number_input("Volatile Acidity")
           citric_acid = st.number_input("Citric Acid")
           residual_sugar = st.number_input("Residual Sugar")
           chlorides = st.number_input("Chlorides")
           free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
    with col2:
           total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
           density = st.number_input("Density")
           pH = st.number_input("pH")
           sulphates = st.number_input("Sulphates")
           alcohol = st.number_input("Alcohol")

    # Create a new dataframe with the input values
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })

    st.write("Given Data:")
    st.table(input_data)
    # Use the trained random forest model to predict the wine quality value
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    X = wine.drop(['quality','Id'] , axis = 1)
    y = wine['quality']
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)

    rf.fit(X_train, y_train)
    prediction = rf.predict(input_data)
    # Display the predicted wine quality value to the user
    st.write("The predicted wine quality is:", prediction[0])
    











     
     