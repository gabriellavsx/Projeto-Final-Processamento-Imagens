import os
import splitfolders
import uuid  # Biblioteca para gerar identificadores únicos

# Caminho das imagens
input_path = 'datasets/'  # Pasta contendo subpastas 'covid' e 'normal'
output_path = 'datasets_splitted/'  # Pasta para salvar as imagens divididas

def rename_images(path):
    if os.path.exists(path):
        for dirpath, _, filenames in os.walk(path):
            for file in filenames:
                full_path = os.path.join(dirpath, file)
                extension = '.' + full_path.split('.')[-1]  # Obtém a extensão do arquivo
                folder_name = os.path.basename(dirpath)

                # Gerando um nome único usando UUID
                newfilename = f"{folder_name}_{uuid.uuid4().hex[:8]}{extension}"
                new_path = os.path.join(dirpath, newfilename)

                os.rename(full_path, new_path)

if __name__ == '__main__':
    rename_images(input_path)
    splitfolders.ratio(input=input_path, output=output_path,
                       seed=1337, ratio=(.8, .2), group_prefix=None, move=False)  # Default values
