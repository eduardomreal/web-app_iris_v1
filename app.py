import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

st.title("Iris prediction web app")

st.sidebar.header("Entrada de valores de parâmetros")

#======== slider ==========
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features
#===========================

df_algum_dado = user_input_features()

st.write('Entrada de valores dos parâmetros')
st.write(df_algum_dado)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

modelo = DecisionTreeClassifier()
modelo.fit(X, Y)

predicao = modelo.predict(df_algum_dado)
predicao_proba = modelo.predict_proba(df_algum_dado)

st.subheader('Calsses e seus índices de identificação')
st.write(iris.target_names)

st.subheader('Predição')
st.write(iris.target_names[predicao])
#st.write(prediction)

st.subheader('Probabilidade de predição')
st.write(predicao_proba)
