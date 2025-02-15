import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Função para carregar os arquivos com tratamento de erro
def load_data(file_path):
    try:
        return np.loadtxt(file_path, delimiter=',')
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar {file_path}: {e}")
        exit(1)  # Interrompe a execução em caso de erro

# Carregar Features e Labels
train_features = load_data('labels/train/features.csv')
train_labels = load_data('labels/train/labels.csv').astype(int)

test_features = load_data('labels/val/features.csv')
test_labels = load_data('labels/val/labels.csv').astype(int)

# Treinando o modelo
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(train_features, train_labels)

# Fazendo previsões
predictions = clf.predict(test_features)

# Criar matriz de confusão
conf_matrix = confusion_matrix(test_labels, predictions)

# Exibir matriz de confusão
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["COVID", "Normal"], yticklabels=["COVID", "Normal"])
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

# Gerar relatório de classificação
report = classification_report(test_labels, predictions, output_dict=True)

# Converter para um DataFrame do Pandas para melhor manipulação
import pandas as pd
df_report = pd.DataFrame(report).T.drop(["accuracy", "macro avg", "weighted avg"])

# Criar gráfico de barras do relatório de classificação
plt.figure(figsize=(8, 5))
df_report[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(8, 5), colormap="viridis")
plt.title("Métricas de Classificação por Classe")
plt.xlabel("Classes")
plt.ylabel("Pontuação")
plt.xticks(rotation=0)
plt.ylim(0, 1)  # Garantir que as métricas fiquem dentro do intervalo 0-1
plt.legend(loc="lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Exibir relatório no terminal
print("Relatório de Classificação:\n", classification_report(test_labels, predictions))
print(f'Accuracy: {accuracy_score(test_labels, predictions)}')
