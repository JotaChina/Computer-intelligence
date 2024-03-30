import re
from unidecode import unidecode

def limpar_arquivo(input_file, output_file):
    # Abre o arquivo de entrada para leitura
    with open(input_file, 'r', encoding='utf-8') as f:
        # Lê todo o conteúdo do arquivo
        texto = f.read()
        
    # Substitui caracteres especiais por seus equivalentes sem acento
    texto_sem_acentos = unidecode(texto)
    
    # Substitui todos os caracteres não alfanuméricos por espaços em branco
    texto_limpo = re.sub(r'[^a-zA-Z0-9\s]', '', texto_sem_acentos)
    
    # Abre o arquivo de saída para escrita
    with open(output_file, 'w', encoding='utf-8') as f:
        # Escreve o texto limpo no arquivo de saída
        f.write(texto_limpo)

# Exemplo de uso
arquivo_entrada = 'lyrics1.txt'
arquivo_saida = 'lyrics.txt'
limpar_arquivo(arquivo_entrada, arquivo_saida)
