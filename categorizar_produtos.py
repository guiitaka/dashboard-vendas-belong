import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import unicodedata
import argparse
import os

def remover_acentos(texto):
    """Remove acentos e caracteres especiais de um texto."""
    if isinstance(texto, str):
        # Normaliza para forma de decomposição e remove os acentos
        texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
        return texto.lower()
    return ""

def preprocessar_texto(texto):
    """Preprocessa o texto para melhorar a correspondência."""
    if not isinstance(texto, str):
        return ""
    
    # Converter para minúsculas e remover acentos
    texto = remover_acentos(texto)
    
    # Remover caracteres especiais e números
    texto = re.sub(r'[^a-zA-Z\s]', ' ', texto)
    
    # Remover espaços extras
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def criar_regras_categorias():
    """
    Cria um dicionário de regras para categorização baseada em palavras-chave.
    Cada categoria tem uma lista de palavras-chave associadas.
    """
    regras = {
        "Maquiagem": [
            # Produtos para lábios
            "batom", "lip", "labial", "gloss", "boca", "labios", "lábios", "lip stick", "lipstick",
            # Base e pó
            "base", "po", "pó", "compacto", "foundation", "bb cream", "cc cream", "corretivo", "concealer",
            "primer", "pre base", "pré base", "fixador", "setting spray", "finalizador",
            # Olhos
            "sombra", "paleta", "palette", "delineador", "eyeliner", "lapis", "lápis", "olho", "eye",
            "rimel", "rímel", "mascara", "máscara", "cilios", "cílios", "sobrancelha", "brow",
            # Face
            "blush", "rouge", "iluminador", "highlighter", "contorno", "bronzer", "bronzeador",
            # Termos gerais
            "maquiagem", "makeup", "make up", "make-up", "cosmetico", "cosmético"
        ],
        
        "Skincare": [
            # Limpeza
            "limpeza", "facial", "demaquilante", "removedor", "sabonete", "gel", "mousse", "espuma",
            "cleansing", "cleanser", "micellar", "micelar", "agua", "água", "tônico", "tonico", "toner",
            # Tratamento
            "serum", "sérum", "ampola", "tratamento", "acido", "ácido", "vitamina c", "retinol",
            "anti-idade", "antiidade", "anti idade", "antirrugas", "anti-rugas", "anti rugas",
            "hidratante", "moisturizer", "creme", "loção", "locao", "gel", "oil free",
            # Proteção
            "protetor solar", "filtro solar", "sunscreen", "fps", "spf", "proteção", "protecao",
            # Esfoliação
            "esfoliante", "peeling", "scrub", "renovador", "renovação", "renovacao",
            # Máscaras
            "mascara facial", "máscara facial", "sheet mask", "mask", "argila", "clay",
            # Termos gerais
            "skincare", "skin care", "pele", "rosto", "face", "facial", "dermatologico", "dermatológico"
        ],
        
        "Cabelo": [
            # Limpeza
            "shampoo", "xampu", "champú", "anti caspa", "anticaspa", "anti-caspa",
            # Condicionamento
            "condicionador", "conditioner", "mascara capilar", "máscara capilar", "hair mask",
            "tratamento", "reparador", "reparação", "reparacao", "reconstrutor", "reconstrução", "reconstrucao",
            # Finalização
            "finalizador", "modelador", "leave-in", "leave in", "creme para pentear", "sem enxague",
            "óleo", "oleo", "serum", "sérum", "spray", "mousse", "espuma", "gel", "pomada",
            # Coloração
            "tintura", "coloração", "coloracao", "color", "tonalizante", "descolorante", "oxidante",
            "matizador", "matizante", "desamarelador",
            # Termos gerais
            "cabelo", "capilar", "hair", "cabeleira", "cabeleireiro", "cabeleireira"
        ],
        
        "Perfumaria": [
            "perfume", "eau de parfum", "eau de toilette", "eau de cologne", "colônia", "colonia",
            "parfum", "fragrance", "fragrância", "fragrancia", "body splash", "body spray",
            "deo parfum", "deo colônia", "deo colonia", "essência", "essencia", "aroma"
        ],
        
        "Corpo": [
            "hidratante corporal", "loção corporal", "locao corporal", "creme corporal", "body lotion",
            "body cream", "manteiga corporal", "óleo corporal", "oleo corporal", "body oil",
            "esfoliante corporal", "body scrub", "sabonete corporal", "body wash", "shower gel",
            "gel de banho", "desodorante", "antitranspirante", "antiperspirante", "desodorante roll-on",
            "desodorante aerosol", "desodorante spray", "talco", "pó corporal", "po corporal",
            "creme para mãos", "creme para maos", "hand cream", "creme para pés", "creme para pes",
            "foot cream", "massagem", "anticelulite", "anti-celulite", "firmador", "redutor de medidas",
            "corpo", "body"
        ],
        
        "Unhas": [
            "esmalte", "nail polish", "base para esmalte", "base coat", "top coat", "finalizador",
            "fortalecedor", "endurecedor", "removedor", "acetona", "cutícula", "cuticula",
            "unha", "unhas", "nail", "nails", "manicure", "pedicure", "alicate", "lixa", "palito"
        ],
        
        "Acessórios": [
            "pincel", "brush", "esponja", "beauty blender", "aplicador", "espátula", "espatula",
            "pente", "escova", "cerdas", "necessaire", "nécessaire", "porta", "estojo", "case",
            "espelho", "mirror", "organizador", "suporte", "kit", "conjunto", "set", "travel",
            "viagem", "bolsa", "sacola", "acessório", "acessorio", "accessory"
        ],
        
        "Cuidados Pessoais": [
            "higiene", "sabonete", "soap", "desodorante", "deodorant", "antitranspirante",
            "antiperspirant", "depilação", "depilacao", "depilador", "cera", "wax", "lâmina",
            "lamina", "barbear", "shaving", "pós-barba", "pos-barba", "after shave", "íntima",
            "intima", "absorvente", "protetor", "lenço", "lenco", "tissue", "papel", "algodão",
            "algodao", "cotonete", "hastes", "escova dental", "creme dental", "pasta de dente",
            "enxaguante", "fio dental", "dental", "oral", "bucal"
        ]
    }
    
    # Preprocessar todas as palavras-chave
    for categoria, palavras in regras.items():
        regras[categoria] = [preprocessar_texto(palavra) for palavra in palavras]
    
    return regras

def categorizar_por_regras(descricao, regras):
    """
    Categoriza um produto com base em regras de palavras-chave.
    
    Args:
        descricao (str): Descrição do produto
        regras (dict): Dicionário de regras de categorização
        
    Returns:
        str: Categoria atribuída ou None se nenhuma correspondência for encontrada
    """
    if not isinstance(descricao, str) or descricao.strip() == "":
        return None
    
    # Preprocessar a descrição
    descricao_prep = preprocessar_texto(descricao)
    
    # Pontuação para cada categoria
    pontuacao_categorias = {}
    
    # Verificar correspondência com cada categoria
    for categoria, palavras_chave in regras.items():
        pontuacao = 0
        for palavra in palavras_chave:
            if palavra in descricao_prep:
                # Palavras exatas têm peso maior
                if f" {palavra} " in f" {descricao_prep} ":
                    pontuacao += 2
                else:
                    pontuacao += 1
        
        if pontuacao > 0:
            pontuacao_categorias[categoria] = pontuacao
    
    # Se encontrou alguma correspondência, retorna a categoria com maior pontuação
    if pontuacao_categorias:
        return max(pontuacao_categorias.items(), key=lambda x: x[1])[0]
    
    return None

def treinar_modelo_similaridade(df, coluna_descricao, coluna_categoria):
    """
    Treina um modelo de similaridade baseado em TF-IDF e KNN.
    
    Args:
        df (DataFrame): DataFrame com os dados
        coluna_descricao (str): Nome da coluna com as descrições
        coluna_categoria (str): Nome da coluna com as categorias
        
    Returns:
        tuple: (vectorizer, modelo, categorias_conhecidas)
    """
    # Filtrar apenas produtos com categorias conhecidas (não vazias e não "Outros")
    df_conhecidos = df[
        (df[coluna_categoria].notna()) & 
        (df[coluna_categoria] != "") & 
        (df[coluna_categoria].str.lower() != "outros")
    ].copy()
    
    if len(df_conhecidos) == 0:
        print("Aviso: Não há produtos com categorias conhecidas para treinar o modelo.")
        return None, None, None
    
    # Preprocessar as descrições
    df_conhecidos['descricao_prep'] = df_conhecidos[coluna_descricao].apply(preprocessar_texto)
    
    # Criar o vetorizador TF-IDF
    vectorizer = TfidfVectorizer(
        min_df=2,           # Ignora termos que aparecem em menos de 2 documentos
        max_df=0.9,         # Ignora termos que aparecem em mais de 90% dos documentos
        ngram_range=(1, 2)  # Considera unigramas e bigramas
    )
    
    # Transformar as descrições em vetores TF-IDF
    X = vectorizer.fit_transform(df_conhecidos['descricao_prep'])
    
    # Treinar o modelo KNN
    modelo = NearestNeighbors(
        n_neighbors=5,      # Considera os 5 vizinhos mais próximos
        metric='cosine'     # Usa similaridade de cosseno
    )
    modelo.fit(X)
    
    # Armazenar as categorias conhecidas
    categorias_conhecidas = df_conhecidos[coluna_categoria].values
    
    return vectorizer, modelo, categorias_conhecidas

def categorizar_por_similaridade(descricao, vectorizer, modelo, categorias_conhecidas):
    """
    Categoriza um produto com base em similaridade de texto.
    
    Args:
        descricao (str): Descrição do produto
        vectorizer: Vetorizador TF-IDF treinado
        modelo: Modelo KNN treinado
        categorias_conhecidas: Array de categorias conhecidas
        
    Returns:
        tuple: (categoria, confiança)
    """
    if not isinstance(descricao, str) or descricao.strip() == "":
        return None, 0.0
    
    # Preprocessar a descrição
    descricao_prep = preprocessar_texto(descricao)
    
    # Transformar a descrição em vetor TF-IDF
    X = vectorizer.transform([descricao_prep])
    
    # Encontrar os vizinhos mais próximos
    distancias, indices = modelo.kneighbors(X)
    
    # Converter distâncias para similaridades (1 - distância)
    similaridades = 1 - distancias[0]
    
    # Obter as categorias dos vizinhos mais próximos
    categorias_vizinhos = [categorias_conhecidas[i] for i in indices[0]]
    
    # Contar a frequência de cada categoria, ponderada pela similaridade
    categoria_scores = {}
    for i, categoria in enumerate(categorias_vizinhos):
        if categoria not in categoria_scores:
            categoria_scores[categoria] = 0
        categoria_scores[categoria] += similaridades[i]
    
    # Encontrar a categoria com maior pontuação
    if categoria_scores:
        categoria_mais_similar = max(categoria_scores.items(), key=lambda x: x[1])
        return categoria_mais_similar[0], categoria_mais_similar[1] / sum(similaridades)
    
    return None, 0.0

def carregar_categorias_referencia(caminho_arquivo):
    """
    Carrega a planilha ou arquivo de categorias de referência.
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo de categorias
        
    Returns:
        dict: Dicionário com mapeamento de categorias
    """
    try:
        mapeamento = {}
        categorias_extraidas = set()
        
        # Verificar a extensão do arquivo
        extensao = os.path.splitext(caminho_arquivo)[1].lower()
        
        # Carregar categorias com base no tipo de arquivo
        if extensao in ['.xlsx', '.xls']:
            # Carregar planilha Excel
            df_categorias = pd.read_excel(caminho_arquivo)
            if len(df_categorias.columns) > 0:
                categorias = df_categorias.iloc[:, 0].dropna().tolist()
        elif extensao == '.csv':
            # Carregar arquivo CSV
            df_categorias = pd.read_csv(caminho_arquivo)
            if len(df_categorias.columns) > 0:
                categorias = df_categorias.iloc[:, 0].dropna().tolist()
        elif extensao == '.md':
            # Carregar arquivo Markdown (uma categoria por linha)
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                categorias = [linha.strip() for linha in f.readlines() if linha.strip()]
        else:
            print(f"Formato de arquivo não suportado: {extensao}")
            return {}
        
        print(f"Carregadas {len(categorias)} categorias do arquivo.")
        
        # Primeiro passo: extrair todas as categorias e subcategorias possíveis
        for categoria in categorias:
            if not isinstance(categoria, str):
                continue
                
            # Normalizar a categoria (remover espaços extras)
            categoria = categoria.strip()
            
            # Verificar se a categoria contém hierarquia (com ">")
            if ">" in categoria:
                # Dividir a hierarquia
                partes = [p.strip() for p in categoria.split(">")]
                
                # Adicionar todas as partes não vazias ao conjunto de categorias
                for parte in partes:
                    if parte.strip() and parte.lower() != "outros":
                        categorias_extraidas.add(parte.strip())
        
        print(f"Extraídas {len(categorias_extraidas)} categorias únicas.")
        
        # Segundo passo: criar mapeamentos para categorias com "Outros"
        for categoria in categorias:
            if not isinstance(categoria, str):
                continue
                
            categoria = categoria.strip()
            
            # Verificar se a categoria contém hierarquia (com ">")
            if ">" in categoria:
                partes = [p.strip() for p in categoria.split(">")]
                
                # Caso 1: Outros > Subcategoria > Categoria Principal
                if partes[0].lower() == "outros" and len(partes) > 1:
                    # Usar a última parte não vazia e não "Outros" como categoria principal
                    for parte in reversed(partes):
                        if parte.strip() and parte.lower() != "outros":
                            mapeamento[categoria.lower()] = parte.strip()
                            break
                
                # Caso 2: Categoria Principal > Subcategoria > Outros
                elif partes[-1].lower() == "outros" and len(partes) > 1:
                    # Priorizar a subcategoria (penúltima parte) se não for "Outros"
                    if len(partes) >= 2 and partes[-2].lower() != "outros":
                        mapeamento[categoria.lower()] = partes[-2].strip()
                    # Caso contrário, usar a primeira parte não "Outros"
                    else:
                        for parte in partes:
                            if parte.lower() != "outros":
                                mapeamento[categoria.lower()] = parte.strip()
                                break
            
            # Caso 3: Categoria simples que termina com "Outros"
            elif categoria.lower().endswith(" outros") and len(categoria) > 7:
                # Remover " Outros" do final
                categoria_principal = categoria[:-7].strip()
                if categoria_principal:  # Se não estiver vazio
                    mapeamento[categoria.lower()] = categoria_principal
        
        # Adicionar mapeamento para a categoria "Outros" isolada
        if "outros" not in mapeamento and categorias_extraidas:
            # Usar uma das categorias extraídas como padrão
            mapeamento["outros"] = list(categorias_extraidas)[0]
        
        print(f"Mapeamento de categorias criado com {len(mapeamento)} entradas")
        # Mostrar algumas amostras do mapeamento para diagnóstico
        amostra = list(mapeamento.items())[:10]
        for k, v in amostra:
            print(f"  '{k}' -> '{v}'")
        
        # Adicionar todas as categorias extraídas ao dicionário de retorno
        resultado = {
            'mapeamento': mapeamento,
            'categorias': list(categorias_extraidas)
        }
        
        return resultado
    except Exception as e:
        print(f"Erro ao carregar arquivo de categorias: {e}")
        import traceback
        traceback.print_exc()
        return {'mapeamento': {}, 'categorias': []}

def categorizar_produtos(df, coluna_descricao, coluna_categoria, limiar_confianca=0.4, arquivo_categorias=None):
    """
    Categoriza produtos com base em regras e similaridade de texto.
    
    Args:
        df (DataFrame): DataFrame com os dados
        coluna_descricao (str): Nome da coluna com as descrições
        coluna_categoria (str): Nome da coluna com as categorias
        limiar_confianca (float): Limiar de confiança para aceitar categorias por similaridade
        arquivo_categorias (str): Caminho para o arquivo de categorias de referência
        
    Returns:
        DataFrame: DataFrame com a nova coluna de categorias corrigidas
    """
    # Criar uma cópia do DataFrame
    df_resultado = df.copy()
    
    # Criar a coluna de categoria corrigida, inicialmente com os valores originais
    df_resultado['categoria_corrigida'] = df_resultado[coluna_categoria]
    
    # Carregar mapeamento de categorias se o arquivo for fornecido
    mapeamento_categorias = {}
    categorias_conhecidas_arquivo = []
    
    if arquivo_categorias and os.path.exists(arquivo_categorias):
        print(f"Carregando mapeamento de categorias de: {arquivo_categorias}")
        resultado_categorias = carregar_categorias_referencia(arquivo_categorias)
        mapeamento_categorias = resultado_categorias['mapeamento']
        categorias_conhecidas_arquivo = resultado_categorias['categorias']
        print(f"Carregado mapeamento de {len(mapeamento_categorias)} categorias.")
        print(f"Categorias conhecidas do arquivo: {categorias_conhecidas_arquivo[:10]}...")
    
    # Aplicar mapeamento de categorias para todas as linhas
    if mapeamento_categorias:
        print("Aplicando mapeamento de categorias...")
        categorias_mapeadas = 0
        
        for idx, row in df_resultado.iterrows():
            categoria_atual = str(row[coluna_categoria]).lower()
            
            # Verificar mapeamento exato
            if categoria_atual in mapeamento_categorias:
                categoria_original = row[coluna_categoria]
                categoria_nova = mapeamento_categorias[categoria_atual]
                df_resultado.at[idx, 'categoria_corrigida'] = categoria_nova
                categorias_mapeadas += 1
                continue
            
            # Verificar se é "Outros" ou contém "Outros"
            if "outros" in categoria_atual:
                # Tentar encontrar a melhor correspondência no mapeamento
                melhor_correspondencia = None
                max_similaridade = 0
                
                for cat_origem, cat_destino in mapeamento_categorias.items():
                    # Calcular similaridade entre strings
                    similaridade = calcular_similaridade(categoria_atual, cat_origem)
                    if similaridade > max_similaridade:
                        max_similaridade = similaridade
                        melhor_correspondencia = cat_destino
                
                # Se encontrou uma correspondência com boa similaridade
                if melhor_correspondencia and max_similaridade > 0.7:
                    df_resultado.at[idx, 'categoria_corrigida'] = melhor_correspondencia
                    categorias_mapeadas += 1
        
        print(f"Total de {categorias_mapeadas} categorias mapeadas diretamente.")
    
    # Criar regras de categorização
    regras = criar_regras_categorias()
    
    # Adicionar categorias conhecidas do arquivo às regras
    for categoria in categorias_conhecidas_arquivo:
        if categoria not in regras:
            # Usar o nome da categoria como palavra-chave
            palavras = preprocessar_texto(categoria).split()
            if palavras:
                if categoria not in regras:
                    regras[categoria] = []
                regras[categoria].extend(palavras)
    
    # Treinar o modelo de similaridade
    vectorizer, modelo, categorias_modelo = treinar_modelo_similaridade(
        df_resultado, coluna_descricao, coluna_categoria
    )
    
    # Identificar produtos que ainda estão como "Outros" ou sem categoria
    mascara_sem_categoria = (
        df_resultado['categoria_corrigida'].isna() | 
        (df_resultado['categoria_corrigida'] == "") | 
        (df_resultado['categoria_corrigida'].str.lower() == "outros") |
        (df_resultado['categoria_corrigida'].str.lower() == "nan")
    )
    
    produtos_sem_categoria = df_resultado[mascara_sem_categoria]
    
    # Contador para estatísticas
    stats = {
        'total': len(produtos_sem_categoria),
        'regras': 0,
        'similaridade': 0,
        'sem_categoria': 0
    }
    
    # Processar cada produto sem categoria
    for idx, row in produtos_sem_categoria.iterrows():
        descricao = row[coluna_descricao]
        
        # Tentar categorizar por regras
        categoria_regras = categorizar_por_regras(descricao, regras)
        
        if categoria_regras:
            df_resultado.at[idx, 'categoria_corrigida'] = categoria_regras
            df_resultado.at[idx, 'metodo_categorizacao'] = 'regras'
            stats['regras'] += 1
        elif vectorizer is not None and modelo is not None:
            # Tentar categorizar por similaridade
            categoria_similaridade, confianca = categorizar_por_similaridade(
                descricao, vectorizer, modelo, categorias_modelo
            )
            
            if categoria_similaridade and confianca >= limiar_confianca:
                df_resultado.at[idx, 'categoria_corrigida'] = categoria_similaridade
                df_resultado.at[idx, 'metodo_categorizacao'] = 'similaridade'
                df_resultado.at[idx, 'confianca_categorizacao'] = confianca
                stats['similaridade'] += 1
            else:
                # Tentar com as categorias conhecidas do arquivo
                melhor_categoria = None
                max_pontuacao = 0
                
                descricao_prep = preprocessar_texto(descricao)
                
                for categoria in categorias_conhecidas_arquivo:
                    # Calcular pontuação baseada na presença de palavras da categoria na descrição
                    palavras_categoria = preprocessar_texto(categoria).split()
                    pontuacao = sum(1 for palavra in palavras_categoria if palavra in descricao_prep)
                    
                    if pontuacao > max_pontuacao:
                        max_pontuacao = pontuacao
                        melhor_categoria = categoria
                
                if melhor_categoria and max_pontuacao > 0:
                    df_resultado.at[idx, 'categoria_corrigida'] = melhor_categoria
                    df_resultado.at[idx, 'metodo_categorizacao'] = 'regras_agressivas'
                    stats['regras'] += 1
                else:
                    # Usar a categoria mais comum como último recurso
                    categorias_comuns = df_resultado['categoria_corrigida'].value_counts()
                    if not categorias_comuns.empty and categorias_comuns.index[0].lower() != "outros":
                        categoria_mais_comum = categorias_comuns.index[0]
                        df_resultado.at[idx, 'categoria_corrigida'] = categoria_mais_comum
                        df_resultado.at[idx, 'metodo_categorizacao'] = 'categoria_mais_comum'
                        stats['similaridade'] += 1
                    else:
                        # Se temos categorias conhecidas do arquivo, usar a primeira
                        if categorias_conhecidas_arquivo:
                            df_resultado.at[idx, 'categoria_corrigida'] = categorias_conhecidas_arquivo[0]
                            df_resultado.at[idx, 'metodo_categorizacao'] = 'categoria_padrao_arquivo'
                            stats['sem_categoria'] += 1
                        else:
                            df_resultado.at[idx, 'categoria_corrigida'] = "Maquiagem"  # Categoria padrão como último recurso
                            df_resultado.at[idx, 'metodo_categorizacao'] = 'sem_correspondencia'
                            stats['sem_categoria'] += 1
        else:
            # Usar a categoria mais comum como último recurso
            categorias_comuns = df_resultado['categoria_corrigida'].value_counts()
            if not categorias_comuns.empty and categorias_comuns.index[0].lower() != "outros":
                categoria_mais_comum = categorias_comuns.index[0]
                df_resultado.at[idx, 'categoria_corrigida'] = categoria_mais_comum
                df_resultado.at[idx, 'metodo_categorizacao'] = 'categoria_mais_comum'
                stats['similaridade'] += 1
            else:
                # Se temos categorias conhecidas do arquivo, usar a primeira
                if categorias_conhecidas_arquivo:
                    df_resultado.at[idx, 'categoria_corrigida'] = categorias_conhecidas_arquivo[0]
                    df_resultado.at[idx, 'metodo_categorizacao'] = 'categoria_padrao_arquivo'
                    stats['sem_categoria'] += 1
                else:
                    df_resultado.at[idx, 'categoria_corrigida'] = "Maquiagem"  # Categoria padrão como último recurso
                    df_resultado.at[idx, 'metodo_categorizacao'] = 'sem_modelo'
                    stats['sem_categoria'] += 1
    
    # Exibir estatísticas
    if stats['total'] > 0:
        print(f"Total de produtos sem categoria após mapeamento: {stats['total']}")
        print(f"Categorizados por regras: {stats['regras']} ({stats['regras']/stats['total']*100:.1f}%)")
        print(f"Categorizados por similaridade: {stats['similaridade']} ({stats['similaridade']/stats['total']*100:.1f}%)")
        print(f"Mantidos como 'Outros': {stats['sem_categoria']} ({stats['sem_categoria']/stats['total']*100:.1f}%)")
    
    # Verificar se ainda existem produtos com categoria "Outros"
    outros_restantes = (df_resultado['categoria_corrigida'].str.lower() == "outros").sum()
    if outros_restantes > 0:
        print(f"ATENÇÃO: Ainda restam {outros_restantes} produtos com categoria 'Outros'")
        
        # Substituir os "Outros" restantes por categorias conhecidas
        if categorias_conhecidas_arquivo:
            categoria_padrao = categorias_conhecidas_arquivo[0]
            print(f"Substituindo 'Outros' restantes por '{categoria_padrao}'")
            df_resultado.loc[df_resultado['categoria_corrigida'].str.lower() == "outros", 'categoria_corrigida'] = categoria_padrao
    
    return df_resultado

# Função auxiliar para calcular similaridade entre strings
def calcular_similaridade(str1, str2):
    """Calcula a similaridade entre duas strings."""
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0
    
    # Normalizar strings
    str1 = str1.lower()
    str2 = str2.lower()
    
    # Calcular similaridade baseada em substrings comuns
    if str1 in str2 or str2 in str1:
        return 0.9
    
    # Calcular similaridade baseada em palavras comuns
    palavras1 = set(str1.split())
    palavras2 = set(str2.split())
    
    # Interseção de palavras
    palavras_comuns = palavras1.intersection(palavras2)
    
    if not palavras1 or not palavras2:
        return 0
    
    # Coeficiente de Jaccard
    return len(palavras_comuns) / (len(palavras1) + len(palavras2) - len(palavras_comuns))

def mapear_categorias_similares(df, coluna_categoria):
    """
    Mapeia categorias menores para categorias principais similares.
    
    Args:
        df (DataFrame): DataFrame com os dados
        coluna_categoria (str): Nome da coluna com as categorias
        
    Returns:
        dict: Dicionário de mapeamento de categorias
    """
    # Obter todas as categorias únicas
    categorias = df[coluna_categoria].unique()
    
    # Categorias principais (as mais frequentes)
    contagem_categorias = df[coluna_categoria].value_counts()
    categorias_principais = contagem_categorias[contagem_categorias > contagem_categorias.mean()].index.tolist()
    
    # Mapeamento de categorias
    mapeamento = {}
    
    # Mapeamentos diretos para casos específicos
    mapeamentos_diretos = {
        'máscara de cílios': 'Maquiagem',
        'mascara de cilios': 'Maquiagem',
        'rímel': 'Maquiagem',
        'rimel': 'Maquiagem',
        'delineador': 'Maquiagem',
        'batom': 'Maquiagem',
        'base': 'Maquiagem',
        'pó': 'Maquiagem',
        'po compacto': 'Maquiagem',
        'blush': 'Maquiagem',
        'primer': 'Maquiagem',
        'corretivo': 'Maquiagem',
        'iluminador': 'Maquiagem',
        'contorno': 'Maquiagem',
        'sombra': 'Maquiagem',
        'paleta': 'Maquiagem',
        'gloss': 'Maquiagem',
        'labial': 'Maquiagem',
        
        'shampoo': 'Cabelos',
        'condicionador': 'Cabelos',
        'máscara capilar': 'Cabelos',
        'mascara capilar': 'Cabelos',
        'tratamento capilar': 'Cabelos',
        'tintura': 'Cabelos',
        'coloração': 'Cabelos',
        'coloracao': 'Cabelos',
        'finalizador': 'Cabelos',
        'modelador': 'Cabelos',
        'gel': 'Cabelos',
        'ativador de cachos': 'Cabelos',
        'creme para pentear': 'Cabelos',
        
        'hidratante facial': 'Skincare',
        'limpeza facial': 'Skincare',
        'tônico': 'Skincare',
        'tonico': 'Skincare',
        'sérum': 'Skincare',
        'serum': 'Skincare',
        'protetor solar': 'Skincare',
        'esfoliante': 'Skincare',
        'máscara facial': 'Skincare',
        'mascara facial': 'Skincare',
        'anti-idade': 'Skincare',
        'antiidade': 'Skincare',
        'acne': 'Skincare',
        
        'perfume': 'Perfumaria',
        'colônia': 'Perfumaria',
        'colonia': 'Perfumaria',
        'eau de parfum': 'Perfumaria',
        'eau de toilette': 'Perfumaria',
        'fragrância': 'Perfumaria',
        'fragrancia': 'Perfumaria',
        
        'sabonete': 'Corpo',
        'hidratante corporal': 'Corpo',
        'loção corporal': 'Corpo',
        'locao corporal': 'Corpo',
        'desodorante': 'Corpo',
        'óleo corporal': 'Corpo',
        'oleo corporal': 'Corpo',
        'esfoliante corporal': 'Corpo',
        
        'esmalte': 'Unhas',
        'base para unhas': 'Unhas',
        'top coat': 'Unhas',
        'acetona': 'Unhas',
        'removedor': 'Unhas',
        
        'pincel': 'Acessórios',
        'escova': 'Acessórios',
        'esponja': 'Acessórios',
        'aplicador': 'Acessórios',
        'necessaire': 'Acessórios',
        'estojo': 'Acessórios'
    }
    
    # Aplicar mapeamentos diretos primeiro
    for categoria in categorias:
        categoria_lower = str(categoria).lower()
        
        # Verificar mapeamentos diretos
        for termo, cat_principal in mapeamentos_diretos.items():
            if termo in categoria_lower:
                mapeamento[categoria] = cat_principal
                break
        
        # Se já foi mapeado, continuar para a próxima categoria
        if categoria in mapeamento:
            continue
        
        # Pular categorias principais
        if categoria in categorias_principais:
            continue
    
    # Palavras-chave para categorias principais (para casos não cobertos pelos mapeamentos diretos)
    palavras_chave_categorias = {
        'Cabelos': ['cabelo', 'capilar', 'shampoo', 'condicionador', 'máscara', 'mascara', 
                   'tratamento', 'hidratante', 'cachos', 'alisamento', 'coloração', 'coloracao',
                   'tintura', 'hair', 'cabeleira', 'cabeleireiro', 'permanente', 'alisante',
                   'relaxante', 'progressiva', 'queratina', 'proteína', 'proteina'],
        
        'Maquiagem': ['batom', 'base', 'pó', 'po', 'blush', 'sombra', 'rímel', 'rimel', 'cílios', 'cilios',
                     'delineador', 'corretivo', 'primer', 'maquiagem', 'makeup', 'labial', 'lábios', 'labios',
                     'gloss', 'contorno', 'iluminador', 'paleta', 'olhos', 'boca', 'face', 'rosto',
                     'sobrancelha', 'brow', 'lash', 'lip', 'eye', 'foundation', 'concealer', 'fixador'],
        
        'Skincare': ['facial', 'rosto', 'pele', 'hidratante', 'limpeza', 'esfoliante', 
                    'tônico', 'tonico', 'sérum', 'serum', 'máscara', 'mascara', 'skincare',
                    'anti-idade', 'antiidade', 'acne', 'protetor solar', 'fps', 'antirrugas',
                    'anti-rugas', 'vitamina c', 'ácido', 'acido', 'hialurônico', 'hialuronico',
                    'retinol', 'peeling', 'demaquilante', 'cleansing', 'toner', 'moisturizer'],
        
        'Perfumaria': ['perfume', 'colônia', 'colonia', 'eau de parfum', 'eau de toilette',
                      'fragrância', 'fragrancia', 'aroma', 'body splash', 'parfum', 'cologne',
                      'deo parfum', 'deo colônia', 'deo colonia', 'essência', 'essencia'],
        
        'Corpo': ['corporal', 'corpo', 'banho', 'sabonete', 'loção', 'locao', 'hidratante',
                 'desodorante', 'óleo', 'oleo', 'esfoliante', 'massagem', 'shower', 'body',
                 'talco', 'pés', 'pes', 'mãos', 'maos', 'hand', 'foot', 'anticelulite',
                 'anti-celulite', 'firmador', 'redutor', 'gel', 'creme'],
        
        'Unhas': ['esmalte', 'unha', 'nail', 'manicure', 'pedicure', 'acetona', 'removedor',
                 'base coat', 'top coat', 'fortalecedor', 'endurecedor', 'cutícula', 'cuticula',
                 'alicate', 'lixa', 'palito', 'polish', 'verniz'],
        
        'Acessórios': ['pincel', 'escova', 'esponja', 'aplicador', 'necessaire', 'estojo',
                      'espelho', 'organizador', 'kit', 'bolsa', 'acessório', 'acessorio',
                      'beauty blender', 'espátula', 'espatula', 'pente', 'cerdas', 'case',
                      'mirror', 'suporte', 'conjunto', 'set', 'travel', 'viagem', 'sacola']
    }
    
    # Para cada categoria não mapeada, verificar correspondência com palavras-chave
    for categoria in categorias:
        if categoria in mapeamento or categoria in categorias_principais:
            continue
            
        categoria_lower = str(categoria).lower()
        melhor_categoria = None
        max_matches = 0
        
        for cat_principal, palavras in palavras_chave_categorias.items():
            # Contar quantas palavras-chave da categoria principal estão na categoria atual
            matches = sum(1 for palavra in palavras if palavra.lower() in categoria_lower)
            
            if matches > max_matches:
                max_matches = matches
                melhor_categoria = cat_principal
        
        # Se encontrou correspondência, mapear para a categoria principal
        if melhor_categoria and max_matches > 0:
            mapeamento[categoria] = melhor_categoria
    
    # Imprimir algumas estatísticas sobre o mapeamento
    print(f"Total de categorias mapeadas: {len(mapeamento)}")
    if mapeamento:
        print("Exemplos de mapeamentos:")
        for i, (cat_orig, cat_dest) in enumerate(list(mapeamento.items())[:5]):
            print(f"  {cat_orig} -> {cat_dest}")
    
    return mapeamento

def main():
    parser = argparse.ArgumentParser(description='Categoriza produtos automaticamente.')
    parser.add_argument('arquivo_entrada', help='Caminho para o arquivo de entrada (CSV, Excel)')
    parser.add_argument('--coluna-descricao', default='Descrição do produto', help='Nome da coluna com as descrições dos produtos')
    parser.add_argument('--coluna-categoria', default='Categoria do produto', help='Nome da coluna com as categorias dos produtos')
    parser.add_argument('--limiar-confianca', type=float, default=0.4, help='Limiar de confiança para aceitar categorias por similaridade')
    parser.add_argument('--arquivo-categorias', help='Caminho para o arquivo de categorias de referência')
    parser.add_argument('--arquivo-saida', help='Caminho para o arquivo de saída (opcional)')
    
    args = parser.parse_args()
    
    # Determinar o formato do arquivo de entrada
    extensao = os.path.splitext(args.arquivo_entrada)[1].lower()
    
    # Carregar o arquivo
    if extensao == '.csv':
        df = pd.read_csv(args.arquivo_entrada)
    elif extensao in ['.xlsx', '.xls']:
        df = pd.read_excel(args.arquivo_entrada)
    else:
        print(f"Formato de arquivo não suportado: {extensao}")
        return
    
    # Verificar se as colunas existem
    if args.coluna_descricao not in df.columns:
        print(f"Coluna de descrição '{args.coluna_descricao}' não encontrada no arquivo.")
        return
    
    if args.coluna_categoria not in df.columns:
        print(f"Coluna de categoria '{args.coluna_categoria}' não encontrada no arquivo.")
        return
    
    # Categorizar os produtos
    df_resultado = categorizar_produtos(
        df, 
        args.coluna_descricao, 
        args.coluna_categoria, 
        args.limiar_confianca,
        args.arquivo_categorias
    )
    
    # Determinar o arquivo de saída
    if args.arquivo_saida:
        arquivo_saida = args.arquivo_saida
    else:
        nome_base, ext = os.path.splitext(args.arquivo_entrada)
        arquivo_saida = f"{nome_base}_categorizado{ext}"
    
    # Salvar o resultado
    if arquivo_saida.endswith('.csv'):
        df_resultado.to_csv(arquivo_saida, index=False)
    else:
        df_resultado.to_excel(arquivo_saida, index=False)
    
    print(f"Arquivo salvo como: {arquivo_saida}")

if __name__ == "__main__":
    main() 