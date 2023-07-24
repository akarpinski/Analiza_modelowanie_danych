#!/usr/bin/env python
# coding: utf-8

# 
# ### Raport z analizy i modelowania wybranego zbioru danych
# Język programowania – Python
# ##### autor: Artur Karpiński

# In[1]:


# Cel badań - przewidywanie zdarzeń chorobowych serca ze względu na zmienną HeartDisease
# HeartDisease  określa stan chorobowy (mediana z danych czterech szpitali) - w skali od 0 (zdrowy) do 1 (chory)

# Metoda badań: klasyfikacja (binarna) - metoda k-najbliższych sąsiadów
# potem porównanie wyników z drzewami decyzyjnymi i lasami losowymi

get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


#ładowanie niezbędnych pakietów
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[3]:


# wczytanie zbioru danych (źródło - https://www.kaggle.com/)
heart = pd.read_csv("dane/heart.csv", comment="#")
heart.head()


# In[4]:


#zbadajmy rozkład liczności klas
ile = heart["HeartDisease"].value_counts()
print(ile)
ile.iloc[np.argsort(ile.index)] #sortowanie


# In[5]:


#dodanie kolumny diagnosis dla poprawy czytelności HeartDisease 
heart["diagnosis"] = pd.cut(heart["HeartDisease"], [0, 1, 2], right=False, labels=["zdrowy", "chory"])
heart["diagnosis"].value_counts()


# In[6]:


heart.dtypes.diagnosis


# In[7]:


heart


# In[8]:


#sprawdzamy typy danych
heart.dtypes


# In[9]:


#zmiana typów danych

#Pierwszy sposób
#Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope są typu object
#musimy je zmienić kolejno na zmienne typu string

#heart.Sex = heart.Sex.astype("string")
#heart.ChestPainType = heart.ChestPainType.astype("string")
#heart.RestingECG = heart.RestingECG.astype("string")
#heart.ExerciseAngina = heart.ExerciseAngina.astype("string")
#heart.ST_Slope = heart.ST_Slope.astype("string")

#heart.dtypes

#Drugi sposób
#Zamieniamy od razu wszystkie definiując zmienną string_col

string_col = heart.select_dtypes(include="object").columns
heart[string_col] = heart[string_col].astype("string")

heart.dtypes


# In[10]:


heart


# In[11]:


#przygotowujemy dane do analizy z 6 wybranych kolumn
#wiek, ciśnienie, cholesterol, poziom cukru [1: BS > 120 mg/dl, 0: inaczej]), tętno
X = heart.iloc[:, [0,3,4,5,7,9]]
X.head()


# In[12]:


y = heart["diagnosis"]
y[0:30]


# In[13]:


#skoro mamy do czynienia z klasyfikacją binarną ("zdrowy" - "chory") - y(i) należy do zbioru {0,1}
#to warto przekodować wartości zmiennej kategorycznej y na zbiór liczb całkowitych
yk = y.cat.codes.values
yk[0:30]


# In[14]:


#kilka kodów badań wybranych losowo
i = np.random.choice(np.arange(len(yk)), 10, replace=False)
yk[i]


# In[15]:


y[i].values


# In[16]:


#Podział zbioru na próbę uczącą i testową
import sklearn.model_selection
np.arange(4)
#np.arange(X.shape[0])


# In[17]:


X.shape[0]


# In[18]:


#wybór indeksów do zbioru treningowego i testowego (funkcja train_test_split)
idx_ucz, idx_test = sklearn.model_selection.train_test_split(np.arange(X.shape[0]),
                                                             test_size=0.2,
                                                             random_state=12345)
X_ucz, X_test = X.iloc[idx_ucz, :], X.iloc[idx_test, :]
y_ucz, y_test = y[idx_ucz], y[idx_test]
yk_ucz, yk_test = yk[idx_ucz], yk[idx_test]
X_ucz.shape, X_test.shape, y_ucz.shape, y_test.shape


# In[19]:


y_ucz.value_counts()


# In[20]:


323/(323+411)


# In[21]:


y_test.value_counts()


# In[22]:


87/(87+97)

#podział na zbiorze testowym trochę odbiega od uczącego, ale do zaakceptowania


# In[23]:


#metoda k-najbliższych sąsiadów
import sklearn.neighbors
knn = sklearn.neighbors.KNeighborsClassifier()
knn.fit(X_ucz, yk_ucz)


# In[24]:


knn.get_params()


# In[25]:


#dla przetestowania skuteczności modelu
#wybieram zbiór przykładowych elementów zbioru testowego
yk_pred = knn.predict(X_test)
yk_pred[[0, 30, 60, 90, 120, 150]]


# In[26]:


#zbiór za pomocą etykiet
k_pred = y.cat.categories[yk_pred]
k_pred[[0, 30, 60, 90, 120, 150]]


# In[27]:


#przykładowe elementy ze zbioru testowego
y_test.values[[0, 30, 60, 90, 120, 150]]

#zgadzają się z oryginalnymi, więc wszystko jest dobrze


# In[28]:


knn.get_params()


# In[29]:


yk_pred_ucz = knn.predict(X_ucz)


# In[30]:


yk_pred = knn.predict(X_test)


# In[31]:


import sklearn.metrics
sklearn.metrics.accuracy_score(yk_test, yk_pred)


# In[32]:


#inny sposób
knn.score(X_test, yk_test)


# In[33]:


sklearn.metrics.accuracy_score(yk_ucz, yk_pred_ucz)


# In[34]:


sklearn.metrics.accuracy_score(yk_test, yk_pred)


# In[35]:


#macierz pomyłek [[true negative, false positive], [false negative, true positive]]
sklearn.metrics.confusion_matrix(yk_test, yk_pred)


# In[36]:


y_test.value_counts()


# In[37]:


from sklearn.metrics import plot_confusion_matrix


# In[38]:


plot_confusion_matrix(knn, X_ucz, yk_ucz)
plt.show()


# In[39]:


plot_confusion_matrix(knn, X_test, yk_test)
plt.show()


# In[40]:


#Metoda trenująca wybrany model alg na zbiorze uczącym (X_ucz, y_ucz), 
#dokonująca predykcji na zbiorze testowym (X_test, y_test)
#walidująca go poprzez cztery wybrane metryki: accuracy, precision, recall i F1
#(ACC, P, R i F1)

def fit_classifier(alg, X_ucz, X_test, y_ucz, y_test):
    alg.fit(X_ucz, y_ucz)
    y_pred = alg.predict(X_test)
    return {
        "ACC": sklearn.metrics.accuracy_score(y_pred, y_test),
        "P":   sklearn.metrics.precision_score(y_pred, y_test),
        "R":   sklearn.metrics.recall_score(y_pred, y_test),
        "F1":  sklearn.metrics.f1_score(y_pred, y_test)
    }


# In[41]:


#stosujemy funkcję fit_classifier dla metryk
pd.Series(fit_classifier(sklearn.neighbors.KNeighborsClassifier(),
                        X_ucz, X_test, yk_ucz, yk_test))


# In[42]:


#tworzymy ramkę danych
params = ["knn"]
res = [fit_classifier(sklearn.neighbors.KNeighborsClassifier(),
                      X_ucz,X_test, yk_ucz, yk_test)]
pd.DataFrame(res, index=params)


# In[43]:


df_results = pd.DataFrame(res, index=params)


# In[44]:


df_results


# In[45]:


X


# In[46]:


#sprawdzamy wyniki dla danych po standaryzacji
m = X.mean()
s = X.std()


# In[47]:


X_ucz_std = (X_ucz - m)/s
X_test_std = (X_test - m)/s


# In[48]:


X_ucz_std.describe()


# In[49]:


params.append("knn_std")
res.append(fit_classifier(sklearn.neighbors.KNeighborsClassifier(),
                          X_ucz_std, X_test_std, yk_ucz, yk_test))
df_results = pd.DataFrame(res, index=params)


# In[50]:


df_results


# In[51]:


#macierz pomyłek
knn.fit(X_ucz_std, yk_ucz)
knn.predict(X_test_std)

plot_confusion_matrix(knn, X_test_std, yk_test)
plt.show()


# In[52]:


yk_pred_test = knn.predict(X_test_std)


# In[53]:


print(sklearn.metrics.classification_report(yk_test, yk_pred_test))


# In[54]:


X_std = (X-m)/s
sns.pairplot(X_std)
plt.show()


# In[55]:


from sklearn.ensemble import IsolationForest


# In[56]:


# fit the model
clf = IsolationForest(random_state=12345)
clf.fit(X)
isf_pred = clf.predict(X)


# In[57]:


len(isf_pred[isf_pred == 1])


# In[58]:


len(isf_pred[isf_pred == -1])


# In[59]:


print(X.shape)
len(isf_pred)


# In[60]:


X_wout_outl = X[isf_pred == 1]


# In[61]:


X_wout_outl


# In[62]:


yk_wout_outl = yk[isf_pred == 1]


# In[63]:


len(yk_wout_outl)


# In[64]:


sns.pairplot(X_wout_outl)
plt.show()


# In[65]:


idx_ucz, idx_test = sklearn.model_selection.train_test_split(np.arange(X_wout_outl.shape[0]),
                                                             test_size=0.2,
                                                             random_state=12345)
X_ucz_wo, X_test_wo = X_wout_outl.iloc[idx_ucz, :], X_wout_outl.iloc[idx_test, :]
yk_ucz_wo, yk_test_wo = yk_wout_outl[idx_ucz], yk_wout_outl[idx_test]
X_ucz_wo.shape, X_test_wo.shape, yk_ucz_wo.shape, yk_test_wo.shape


# In[66]:


X_test_std


# In[67]:


tab_train = list()
tab_test = list()

for i in range(1,31):
    klasyfikator = sklearn.neighbors.KNeighborsClassifier(n_neighbors=i) #tworzenie modelu
    print(klasyfikator)
    klasyfikator.fit(X_ucz_std, yk_ucz) #trenowanie modelu
    
    Y_tr_pred = klasyfikator.predict(X_ucz_std)
    Y_pred = klasyfikator.predict(X_test_std) #klasyfikacja zmiennej celu dla zbioru testowego
    
    tab_train.append(sklearn.metrics.f1_score(yk_ucz, Y_tr_pred))
    tab_test.append(sklearn.metrics.f1_score(yk_test, Y_pred))


# In[68]:


#poniżej wynik działania pętli od n=1 do wybranej maksymalnej liczby 30 sąsiadów
plt.figure(figsize=(14,7))
plt.plot(tab_train, label='train')
plt.plot(tab_test, label='test')
plt.legend()
plt.show()


# In[69]:


#czyli widzimy, że najlepsze wyniki przy liczbie 26 sąsiadów (n_neighbors=26)
#między 17, a 26 się przecinają w różnych punktach
#potem się rozchodzą na stałe, czyli model już się nie uczy

params.append("knn26")
res.append(fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=26),
                          X_ucz, X_test, yk_ucz, yk_test))
df_results = pd.DataFrame(res, index=params)


# In[70]:


df_results


# In[71]:


#sprawdzamy wyniki dla danych po standaryzacji
params.append("knn26_std")
res.append(fit_classifier(sklearn.neighbors.KNeighborsClassifier(n_neighbors=26),
                          X_ucz_std, X_test_std, yk_ucz, yk_test))
df_results = pd.DataFrame(res, index=params)


# In[72]:


df_results


# In[73]:


#mieliśmy macierz pomyłek na knn
knn.fit(X_ucz_std, yk_ucz)
knn.predict(X_test_std)

plot_confusion_matrix(knn, X_test_std, yk_test)
plt.show()


# In[74]:


#teraz na knn26
knn26 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=26)
knn26.fit(X_ucz_std, yk_ucz)
knn26.predict(X_test_std)

plot_confusion_matrix(knn26, X_test_std, yk_test)
plt.show()
#widzimy na knn26 mniej błędów 1 i 2 rodzaju niż w knn


# In[75]:


#na koniec sprawdzimy jeszcze drzewa decyzyjne i lasy losowe
import sklearn.tree


# In[76]:


#drzewa decyzyjne (bez określonej maksymalnej głębokości)
params.append("dt")
res.append(fit_classifier(sklearn.tree.DecisionTreeClassifier(),
                          X_ucz, X_test, yk_ucz, yk_test))
df_results = pd.DataFrame(res, index=params)


# In[77]:


df_results


# In[78]:


moje_drzewo = sklearn.tree.DecisionTreeClassifier()
#parametry algorytmu
moje_drzewo.fit(X_ucz, yk_ucz)


# In[79]:


moje_drzewo.get_params()


# In[80]:


#sprawdźmy głębokość drzewa 
moje_drzewo.get_depth()


# In[81]:


#czyli domyślna głębokość drzewa wyszła 23 
#weźmy konkretną głębokość drzewa 12
params.append("dt_maxd12")
res.append(fit_classifier(sklearn.tree.DecisionTreeClassifier(max_depth=12),
                          X_ucz, X_test, yk_ucz, yk_test))
df_results = pd.DataFrame(res, index=params)


# In[82]:


df_results
#poniżej widzimy, że przy nie określonym max_depth wyniki słabe,
#ale przy głębokości drzewa max_depth=12 już lepsze


# In[83]:


#sprawdzamy wyniki dla danych po standaryzacji
params.append("dt_maxd12_std")
res.append(fit_classifier(sklearn.tree.DecisionTreeClassifier(max_depth=12),
                          X_ucz_std, X_test_std, yk_ucz, yk_test))
df_results = pd.DataFrame(res, index=params)


# In[84]:


df_results


# In[85]:


#czyli na drzewach decyzyjnych, również z ustaloną głębokością max_depth=12
#wyniki gorsze niż na metodzie k-najbliższych sąsiadów (26 sąsiadów)


# In[86]:


#lasy losowe
import sklearn.ensemble


# In[87]:


params.append("rf")
res.append(fit_classifier(sklearn.ensemble.RandomForestClassifier(random_state=12345),
                          X_ucz, X_test, yk_ucz, yk_test))
df_results = pd.DataFrame(res, index=params)


# In[88]:


df_results


# In[89]:


import sklearn.ensemble
las = sklearn.ensemble.RandomForestClassifier(random_state=12345)
#parametry algorytmu
las.fit(X_ucz, yk_ucz)


# In[90]:


#przy ustalonej głębokości max_depth=12
params.append("rf_maxd12_oob")
res.append(fit_classifier(sklearn.ensemble.RandomForestClassifier(max_depth=12,
                                                                  oob_score=True,
                                                                  random_state=12345),
                          X_ucz, X_test, yk_ucz, yk_test))
df_results = pd.DataFrame(res, index=params)


# In[91]:


df_results


# In[92]:


#po standardyzacji
params.append("rf_maxd12_oob_std")
res.append(fit_classifier(sklearn.ensemble.RandomForestClassifier(max_depth=12,
                                                                  oob_score=True,
                                                                  random_state=12345),
                          X_ucz_std, X_test_std, yk_ucz, yk_test))
df_results = pd.DataFrame(res, index=params)


# In[93]:


df_results
#czyli na lasach losowych (też z max_depth=12) również wyniki lepsze
#niż na metodzie k-najbliższych sąsiadów (26 sąsiadów)


# ###  Wniosek końcowy:
# Na pobranych danych analizowanych za pomocą 3 metod optymalne wyniki otrzymaliśmy chyba na metodzie lasów losowych.
# Ale mogą odbiegać od rzeczywistości, gdyż wybrałem jedynie kilka z podanych kolumn.
# 
# Jeśli je dobrze zinterpretowałem.

# In[ ]:




