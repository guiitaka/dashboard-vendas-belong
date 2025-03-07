import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from utils import process_data, generate_insights
from categorizar_produtos import categorizar_produtos, criar_regras_categorias, categorizar_por_regras, treinar_modelo_similaridade, categorizar_por_similaridade, mapear_categorias_similares
import os
import json
from urllib.parse import quote
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import tempfile

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Vendas - Marketplace",
    page_icon="📊",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
<style>
.download-button {
    display: inline-block;
    padding: 0.5em 1em;
    background-color: #4CAF50;
    color: white;
    text-align: center;
    text-decoration: none;
    font-size: 16px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    margin-top: 10px;
}
.download-button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# Adicione apenas o título
st.title("📊 Dashboard de Análise de Vendas")
st.markdown("---")

# Área de upload de arquivo
st.sidebar.header("Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Faça upload da planilha de vendas", type=["xlsx", "csv"])

# Adicione esta opção para usar dados de exemplo
use_example_data = st.sidebar.checkbox("Usar dados de exemplo", False)

# Função para carregar os dados
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    # Renomear colunas para garantir consistência
    column_mapping = {
        'Número': 'numero_pedido',
        'Data da venda': 'data_venda',
        'Quantidade de produtos': 'quantidade',
        'Valor total da venda': 'valor_total',
        'Categoria do produto': 'categoria',
        'Descrição do produto': 'descricao'
    }
    
    # Tentar mapear as colunas existentes
    for original, novo in column_mapping.items():
        if original in df.columns:
            df = df.rename(columns={original: novo})
    
    # Garantir que a coluna de data está no formato correto
    if 'data_venda' in df.columns:
        df['data_venda'] = pd.to_datetime(df['data_venda'])
    
    # Converter a coluna categoria para string
    if 'categoria' in df.columns:
        df['categoria'] = df['categoria'].astype(str)
    
    # Verificar se temos a coluna de descrição e a coluna de categoria
    if 'descricao' in df.columns and 'categoria' in df.columns:
        # Procurar por arquivos de categorias em várias extensões
        arquivos_possiveis = [
            os.path.join(os.path.dirname(__file__), "categorias-produtos.md"),
            os.path.join(os.path.dirname(__file__), "categorias-produtos.xls"),
            os.path.join(os.path.dirname(__file__), "categorias-produtos.xlsx"),
            os.path.join(os.path.dirname(__file__), "categorias-produtos.csv")
        ]
        
        arquivo_categorias = None
        for arquivo in arquivos_possiveis:
            if os.path.exists(arquivo):
                arquivo_categorias = arquivo
                break
        
        # Mostrar um spinner enquanto categoriza os produtos
        with st.spinner('Categorizando produtos automaticamente...'):
            # Categorizar produtos sem categoria ou com categoria "Outros"
            df_categorizado = categorizar_produtos(
                df, 
                coluna_descricao='descricao', 
                coluna_categoria='categoria', 
                limiar_confianca=0.4,
                arquivo_categorias=arquivo_categorias
            )
            
            # Usar a categoria corrigida em vez da original
            df_categorizado['categoria'] = df_categorizado['categoria_corrigida']
            
            # Remover colunas temporárias usadas na categorização
            colunas_para_remover = ['categoria_corrigida', 'metodo_categorizacao', 'confianca_categorizacao']
            for col in colunas_para_remover:
                if col in df_categorizado.columns:
                    df_categorizado = df_categorizado.drop(col, axis=1)
            
            # Mapear categorias similares para reduzir a categoria "Outros"
            with st.spinner('Otimizando categorias...'):
                # Guardar a categoria original antes do mapeamento
                df_categorizado['categoria_original'] = df_categorizado['categoria']
                
                # Obter mapeamento de categorias similares
                mapeamento_categorias = mapear_categorias_similares(df_categorizado, 'categoria')
                
                # Aplicar mapeamento
                df_categorizado['categoria'] = df_categorizado['categoria'].apply(
                    lambda x: mapeamento_categorias.get(x, x)
                )
                
                # Mostrar informações sobre o mapeamento
                categorias_mapeadas = len(set(mapeamento_categorias.keys()))
                if categorias_mapeadas > 0:
                    st.sidebar.success(f"✅ {categorias_mapeadas} categorias menores foram mapeadas para categorias principais!")
            
            return df_categorizado
    
    return df

# Adicione esta função simplificada para exportar para CSV
def get_csv_download_link(df, filename="relatorio_vendas.csv"):
    """
    Gera um link HTML para download dos dados em formato CSV.
    
    Args:
        df (DataFrame): DataFrame com os dados
        filename (str): Nome do arquivo para download
        
    Returns:
        str: Link HTML para download
    """
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    b64 = base64.b64encode(csv_buffer.read()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">📥 Baixar Dados CSV</a>'
    return href

# Função para criar um link de download
def get_pdf_download_link(pdf_bytes, filename="relatorio_vendas.pdf"):
    """
    Gera um link HTML para download do PDF.
    
    Args:
        pdf_bytes (BytesIO): Buffer contendo o PDF
        filename (str): Nome do arquivo para download
        
    Returns:
        str: Link HTML para download
    """
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-button">📥 Baixar Relatório PDF</a>'
    return href

# Adicione esta função para criar o PDF
def create_pdf_report(df):
    """
    Cria um relatório PDF com os principais insights e gráficos do dashboard.
    
    Args:
        df (DataFrame): DataFrame com os dados
        
    Returns:
        BytesIO: Buffer contendo o PDF
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Estilos
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Estilo personalizado para itens de lista
    list_style = ParagraphStyle(
        'ListItem',
        parent=styles['Normal'],
        leftIndent=20,
        firstLineIndent=0,
        spaceBefore=2,
        spaceAfter=2
    )
    
    # Conteúdo do PDF
    content = []
    
    # Título e data
    title_style.alignment = 1  # Centralizado
    content.append(Paragraph("Relatório de Análise de Vendas", title_style))
    
    date_style = ParagraphStyle('Date', parent=styles['Normal'], alignment=1)
    content.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", date_style))
    content.append(Spacer(1, 0.5*inch))
    
    # Resumo dos dados
    content.append(Paragraph("Resumo dos Dados", subtitle_style))
    
    # Métricas principais
    total_vendas = df['valor_total'].sum()
    total_produtos = df['quantidade'].sum() if 'quantidade' in df.columns else 0
    total_pedidos = df['numero_pedido'].nunique() if 'numero_pedido' in df.columns else 0
    ticket_medio = total_vendas / total_pedidos if total_pedidos > 0 else 0
    
    # Período de análise
    if 'data_venda' in df.columns:
        data_inicio = df['data_venda'].min().strftime('%d/%m/%Y')
        data_fim = df['data_venda'].max().strftime('%d/%m/%Y')
        periodo = f"Período analisado: {data_inicio} a {data_fim}"
    else:
        periodo = "Período analisado: Não disponível"
    
    metricas_texto = f"""
    • {periodo}
    • Total de Vendas: R$ {total_vendas:,.2f}
    • Total de Produtos Vendidos: {total_produtos:,}
    • Total de Pedidos: {total_pedidos:,}
    • Ticket Médio: R$ {ticket_medio:,.2f}
    """
    content.append(Paragraph(metricas_texto, normal_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Vendas por Categoria
    content.append(Paragraph("Vendas por Categoria", subtitle_style))
    
    # Criar gráfico de vendas por categoria
    vendas_categoria = df.groupby('categoria')['valor_total'].sum().reset_index()
    vendas_categoria = vendas_categoria.sort_values('valor_total', ascending=False)
    
    # Limitar a 10 categorias para melhor visualização
    if len(vendas_categoria) > 10:
        top_categorias = vendas_categoria.head(9)
        outras_categorias = pd.DataFrame({
            'categoria': ['Outras'],
            'valor_total': [vendas_categoria.iloc[9:]['valor_total'].sum()]
        })
        vendas_categoria_plot = pd.concat([top_categorias, outras_categorias])
    else:
        vendas_categoria_plot = vendas_categoria
    
    # Salvar gráfico em um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        plt.figure(figsize=(8, 5))
        plt.bar(vendas_categoria_plot['categoria'], vendas_categoria_plot['valor_total'], color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title('Vendas por Categoria')
        plt.ylabel('Valor Total (R$)')
        plt.tight_layout()
        plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Adicionar gráfico ao PDF
        img = Image(temp_file.name, width=6*inch, height=4*inch)
        content.append(img)
        content.append(Spacer(1, 0.25*inch))
    
    # Tabela de top produtos
    content.append(Paragraph("Top 10 Produtos Mais Vendidos", subtitle_style))
    
    if 'descricao' in df.columns:
        top_produtos = df.groupby('descricao').agg({
            'valor_total': 'sum',
            'quantidade': 'sum' if 'quantidade' in df.columns else 'count'
        }).sort_values('valor_total', ascending=False).head(10).reset_index()
        
        produtos_texto = ""
        for i, row in top_produtos.iterrows():
            desc = row['descricao']
            if len(desc) > 50:
                desc = desc[:47] + "..."
            
            qtd = row['quantidade']
            produtos_texto += f"{i+1}. {desc} - R$ {row['valor_total']:,.2f} ({qtd} unidades)\n"
        
        content.append(Paragraph(produtos_texto, normal_style))
        content.append(Spacer(1, 0.25*inch))
    
    # Insights simplificados (sem gráficos Plotly)
    content.append(Paragraph("Principais Insights", subtitle_style))
    
    # Criar insights simplificados sem depender de objetos Plotly
    insights_texto = ""
    
    # Insight 1: Categoria mais vendida
    top_categoria = vendas_categoria.iloc[0]['categoria']
    valor_top = vendas_categoria.iloc[0]['valor_total']
    percentual = (valor_top / total_vendas) * 100
    insights_texto += f"• A categoria mais vendida é '{top_categoria}', representando {percentual:.1f}% do total (R$ {valor_top:,.2f})\n\n"
    
    # Insight 2: Crescimento de vendas (se houver dados temporais)
    if 'data_venda' in df.columns:
        df_tempo = df.copy()
        df_tempo['mes_ano'] = df_tempo['data_venda'].dt.strftime('%Y-%m')
        vendas_tempo = df_tempo.groupby('mes_ano')['valor_total'].sum().reset_index()
        
        if len(vendas_tempo) > 1:
            primeiro_periodo = vendas_tempo.iloc[0]['valor_total']
            ultimo_periodo = vendas_tempo.iloc[-1]['valor_total']
            variacao = ((ultimo_periodo / primeiro_periodo) - 1) * 100
            
            if variacao > 0:
                insights_texto += f"• Crescimento de {variacao:.1f}% nas vendas entre o primeiro e o último período\n\n"
            else:
                insights_texto += f"• Redução de {abs(variacao):.1f}% nas vendas entre o primeiro e o último período\n\n"
    
    # Insight 3: Ticket médio por categoria
    ticket_medio_cat = df.groupby('categoria').agg({
        'valor_total': 'sum',
        'numero_pedido': 'nunique'
    })
    ticket_medio_cat['ticket_medio'] = ticket_medio_cat['valor_total'] / ticket_medio_cat['numero_pedido']
    ticket_medio_cat = ticket_medio_cat.sort_values('ticket_medio', ascending=False)
    
    if not ticket_medio_cat.empty:
        top_ticket = ticket_medio_cat.iloc[0]
        insights_texto += f"• A categoria com maior ticket médio é '{ticket_medio_cat.index[0]}' (R$ {top_ticket['ticket_medio']:,.2f})\n\n"
    
    # Insight 4: Produtos mais vendidos
    if 'descricao' in df.columns and 'quantidade' in df.columns:
        top_qtd = df.groupby('descricao')['quantidade'].sum().sort_values(ascending=False)
        if not top_qtd.empty:
            insights_texto += f"• O produto mais vendido em quantidade é '{top_qtd.index[0]}' ({top_qtd.iloc[0]} unidades)\n\n"
    
    # Insight 5: Dia da semana com mais vendas
    if 'data_venda' in df.columns:
        df['dia_semana'] = df['data_venda'].dt.day_name()
        vendas_dia = df.groupby('dia_semana')['valor_total'].sum().reset_index()
        
        # Ordenar dias da semana
        dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_pt = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
        dias_map = dict(zip(dias_ordem, dias_pt))
        
        vendas_dia['dia_pt'] = vendas_dia['dia_semana'].map(dias_map)
        vendas_dia = vendas_dia.sort_values('valor_total', ascending=False)
        
        if not vendas_dia.empty:
            melhor_dia = vendas_dia.iloc[0]['dia_pt']
            pior_dia = vendas_dia.iloc[-1]['dia_pt']
            insights_texto += f"• O melhor dia para vendas é {melhor_dia}, enquanto {pior_dia} apresenta o menor volume\n"
    
    content.append(Paragraph(insights_texto, normal_style))
    
    # Adicionar informações sobre categorização
    if 'categoria_original' in df.columns:
        content.append(Spacer(1, 0.25*inch))
        content.append(Paragraph("Informações sobre Categorização", subtitle_style))
        
        # Contar quantos produtos foram recategorizados
        recategorizados = df[df['categoria'] != df['categoria_original']].shape[0]
        total_produtos = df.shape[0]
        percentual = (recategorizados / total_produtos) * 100 if total_produtos > 0 else 0
        
        cat_texto = f"""
        • {recategorizados} produtos ({percentual:.1f}%) foram recategorizados automaticamente
        • A categorização automática melhora a qualidade da análise agrupando produtos similares
        """
        content.append(Paragraph(cat_texto, normal_style))
    
    # Rodapé
    content.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], alignment=1, fontSize=8, textColor=colors.gray)
    content.append(Paragraph("Dashboard de Análise de Vendas | Gerado automaticamente", footer_style))
    
    # Construir o PDF
    doc.build(content)
    
    # Limpar arquivos temporários
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)
    
    buffer.seek(0)
    return buffer

# Adicione o botão de exportação CSV no sidebar
if use_example_data:
    # Carregar dados de exemplo
    @st.cache_data
    def load_example_data():
        # Criar um DataFrame de exemplo ou carregar de um arquivo incluído no repositório
        df_example = pd.DataFrame({
            'numero_pedido': range(1000, 1100),
            'data_venda': pd.date_range(start='2023-01-01', periods=100),
            'quantidade': np.random.randint(1, 10, size=100),
            'valor_total': np.random.uniform(50, 500, size=100),
            'categoria': np.random.choice(['Maquiagem', 'Cabelos', 'Skincare', 'Perfumaria', 'Corpo'], size=100),
            'descricao': ['Produto ' + str(i) for i in range(100)]
        })
        return df_example
    
    df = load_example_data()
    df_processed = process_data(df)
    
    # Mostrar mensagem informativa
    st.sidebar.success("Usando dados de exemplo. Faça upload de seus próprios dados para análise personalizada.")
    
elif uploaded_file is not None:
    # Carregar e processar os dados
    with st.spinner('Carregando e processando dados...'):
        df = load_data(uploaded_file)
        df_processed = process_data(df)
    
    # Adicione o botão de exportação CSV
    if st.sidebar.button("📥 Exportar Relatório", type="primary"):
        try:
            with st.spinner("Gerando relatório CSV..."):
                # Gerar CSV
                csv_buffer = BytesIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                # Criar link de download
                b64 = base64.b64encode(csv_buffer.read()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="relatorio_vendas.csv" class="download-button">📥 Baixar Dados CSV</a>'
                st.sidebar.markdown(href, unsafe_allow_html=True)
                st.sidebar.success("Relatório CSV gerado com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao gerar relatório: {str(e)}")
    
    # Exibir informações sobre a categorização automática
    if 'descricao' in df.columns:
        st.sidebar.success("✅ Categorização automática aplicada com sucesso!")
        st.sidebar.info("""
        **Nota:** Produtos sem categoria ou classificados como "Outros" foram 
        automaticamente categorizados com base na descrição do produto.
        """)
    
    # Exibir informações básicas
    st.sidebar.success(f"Arquivo carregado com sucesso: {uploaded_file.name}")
    st.sidebar.info(f"Total de registros: {len(df)}")
    st.sidebar.info(f"Período: {df['data_venda'].min().strftime('%d/%m/%Y')} a {df['data_venda'].max().strftime('%d/%m/%Y')}")
    
    # Layout do dashboard em abas
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Análise Temporal", "Análise por Categoria", "Insights"])
    
    with tab1:
        st.header("Visão Geral das Vendas")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_vendas = df['valor_total'].sum()
            st.metric("Total de Vendas", f"R$ {total_vendas:,.2f}")
        
        with col2:
            total_pedidos = df['numero_pedido'].nunique()
            st.metric("Total de Pedidos", f"{total_pedidos:,}")
        
        with col3:
            total_produtos = df['quantidade'].sum()
            st.metric("Produtos Vendidos", f"{total_produtos:,}")
        
        with col4:
            ticket_medio = total_vendas / total_pedidos if total_pedidos > 0 else 0
            st.metric("Ticket Médio", f"R$ {ticket_medio:,.2f}")
        
        # Gráfico de vendas por categoria
        st.subheader("Vendas por Categoria")
        vendas_categoria = df.groupby('categoria')['valor_total'].sum().reset_index()
        vendas_categoria = vendas_categoria.sort_values('valor_total', ascending=False)

        # Agrupar categorias pequenas em "Outros"
        limite_percentual = 2.0  # Categorias com menos de 2% serão agrupadas em "Outros"
        total_vendas = vendas_categoria['valor_total'].sum()
        vendas_categoria['percentual'] = (vendas_categoria['valor_total'] / total_vendas) * 100

        # Separar categorias principais e secundárias
        categorias_principais = vendas_categoria[vendas_categoria['percentual'] >= limite_percentual]
        categorias_secundarias = vendas_categoria[vendas_categoria['percentual'] < limite_percentual]

        # Criar categoria "Outros" se houver categorias secundárias
        if not categorias_secundarias.empty:
            outros = pd.DataFrame({
                'categoria': ['Outros'],
                'valor_total': [categorias_secundarias['valor_total'].sum()],
                'percentual': [categorias_secundarias['percentual'].sum()]
            })
            vendas_categoria_final = pd.concat([categorias_principais, outros])
        else:
            vendas_categoria_final = categorias_principais

        # Ordenar por valor para melhor visualização
        vendas_categoria_final = vendas_categoria_final.sort_values('valor_total', ascending=False)

        # Criar gráfico de pizza melhorado
        fig_cat = px.pie(
            vendas_categoria_final, 
            values='valor_total', 
            names='categoria',
            title='Distribuição de Vendas por Categoria',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        # Melhorar a formatação do gráfico
        fig_cat.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextorientation='radial',
            hovertemplate='<b>%{label}</b><br>Valor: R$ %{value:,.2f}<br>Percentual: %{percent:.1%}<extra></extra>'
        )

        # Melhorar o layout
        fig_cat.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=60, b=120, l=20, r=20)
        )

        # Criar um container para o gráfico e o modal
        chart_container = st.container()

        with chart_container:
            # Renderizar o gráfico
            st.plotly_chart(fig_cat, use_container_width=True)
            
            # Adicionar seletor de categoria para mostrar detalhes
            st.markdown("### Detalhes por Categoria")
            categorias_disponiveis = vendas_categoria_final['categoria'].tolist()
            categoria_selecionada = st.selectbox(
                "Selecione uma categoria para ver os produtos:",
                options=categorias_disponiveis
            )
            
            # Mostrar produtos da categoria selecionada
            if categoria_selecionada:
                # Verificar se é a categoria "Outros" (agrupada)
                if categoria_selecionada == "Outros":
                    # Obter todas as categorias pequenas que foram agrupadas em "Outros"
                    categorias_pequenas = categorias_secundarias['categoria'].tolist()
                    
                    # Filtrar produtos de todas as categorias pequenas
                    category_products = df[df['categoria'].isin(categorias_pequenas)].copy()
                    
                    # Mostrar quais categorias foram agrupadas
                    st.info(f"A categoria 'Outros' agrupa {len(categorias_pequenas)} categorias menores: {', '.join(categorias_pequenas[:10])}{'...' if len(categorias_pequenas) > 10 else ''}")
                else:
                    # Para outras categorias, filtrar normalmente
                    category_products = df[df['categoria'] == categoria_selecionada].copy()
                
                # Mostrar informações sobre a categoria
                st.write(f"**Total de produtos na categoria '{categoria_selecionada}':** {len(category_products)}")
                
                # Mostrar valor total da categoria
                valor_categoria = category_products['valor_total'].sum()
                percentual = (valor_categoria / total_vendas) * 100
                st.write(f"**Valor total:** R$ {valor_categoria:,.2f} ({percentual:.1f}% do total)")
                
                # Mostrar tabela de produtos
                if len(category_products) > 0:
                    st.write("### Lista de Produtos")
                    
                    # Se for a categoria "Outros", adicionar a coluna de categoria original
                    if categoria_selecionada == "Outros":
                        if 'descricao' in category_products.columns:
                            product_table = category_products[['descricao', 'categoria', 'valor_total']].copy()
                            product_table = product_table.rename(columns={
                                'descricao': 'Descrição do Produto',
                                'categoria': 'Categoria Original',
                                'valor_total': 'Valor Total (R$)'
                            })
                        else:
                            product_table = category_products[['categoria', 'valor_total']].copy()
                            product_table = product_table.rename(columns={
                                'categoria': 'Categoria Original',
                                'valor_total': 'Valor Total (R$)'
                            })
                    else:
                        # Para categorias principais, mostrar a categoria original se disponível
                        if 'categoria_original' in category_products.columns:
                            product_table = category_products[['descricao', 'categoria_original', 'valor_total']].copy()
                            product_table = product_table.rename(columns={
                                'descricao': 'Descrição do Produto',
                                'categoria_original': 'Categoria Original',
                                'valor_total': 'Valor Total (R$)'
                            })
                        elif 'descricao' in category_products.columns:
                            product_table = category_products[['descricao', 'valor_total']].copy()
                            product_table = product_table.rename(columns={
                                'descricao': 'Descrição do Produto',
                                'valor_total': 'Valor Total (R$)'
                            })
                        else:
                            product_table = category_products[['valor_total']].copy()
                            product_table = product_table.rename(columns={
                                'valor_total': 'Valor Total (R$)'
                            })
                    
                    product_table['Valor Total (R$)'] = product_table['Valor Total (R$)'].map('R$ {:.2f}'.format)
                    st.dataframe(product_table, use_container_width=True)
                else:
                    st.warning("Não foram encontrados produtos para esta categoria.")
        
        # Gráfico de quantidade de produtos por categoria
        st.subheader("Quantidade de Produtos por Categoria")
        qtd_categoria = df.groupby('categoria')['quantidade'].sum().reset_index()
        fig_qtd = px.bar(
            qtd_categoria,
            x='categoria',
            y='quantidade',
            title='Quantidade de Produtos Vendidos por Categoria',
            color='categoria',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_qtd, use_container_width=True)
    
    with tab2:
        st.header("Análise Temporal")
        
        # Agregação por período
        periodo_options = ["Diário", "Semanal", "Mensal"]
        periodo_selecionado = st.selectbox("Selecione o período de análise:", periodo_options)
        
        if periodo_selecionado == "Diário":
            df_tempo = df.groupby(df['data_venda'].dt.date).agg({
                'valor_total': 'sum',
                'quantidade': 'sum',
                'numero_pedido': 'nunique'
            }).reset_index()
            x_axis = 'data_venda'
        elif periodo_selecionado == "Semanal":
            df['semana'] = df['data_venda'].dt.isocalendar().week
            df['ano'] = df['data_venda'].dt.isocalendar().year
            df['semana_ano'] = df['ano'].astype(str) + "-" + df['semana'].astype(str)
            df_tempo = df.groupby('semana_ano').agg({
                'valor_total': 'sum',
                'quantidade': 'sum',
                'numero_pedido': 'nunique'
            }).reset_index()
            x_axis = 'semana_ano'
        else:  # Mensal
            df['mes_ano'] = df['data_venda'].dt.strftime('%Y-%m')
            df_tempo = df.groupby('mes_ano').agg({
                'valor_total': 'sum',
                'quantidade': 'sum',
                'numero_pedido': 'nunique'
            }).reset_index()
            x_axis = 'mes_ano'
        
        # Gráfico de linha para vendas ao longo do tempo
        st.subheader(f"Evolução de Vendas ({periodo_selecionado})")
        fig_tempo = px.line(
            df_tempo,
            x=x_axis,
            y='valor_total',
            markers=True,
            title=f'Evolução do Valor Total de Vendas ({periodo_selecionado})',
            labels={'valor_total': 'Valor Total (R$)', x_axis: 'Período'}
        )
        st.plotly_chart(fig_tempo, use_container_width=True)
        
        # Gráfico de barras para quantidade de produtos ao longo do tempo
        st.subheader(f"Evolução da Quantidade de Produtos ({periodo_selecionado})")
        fig_qtd_tempo = px.bar(
            df_tempo,
            x=x_axis,
            y='quantidade',
            title=f'Evolução da Quantidade de Produtos Vendidos ({periodo_selecionado})',
            labels={'quantidade': 'Quantidade', x_axis: 'Período'}
        )
        st.plotly_chart(fig_qtd_tempo, use_container_width=True)
        
        # Gráfico de linha para número de pedidos ao longo do tempo
        st.subheader(f"Evolução do Número de Pedidos ({periodo_selecionado})")
        fig_pedidos = px.line(
            df_tempo,
            x=x_axis,
            y='numero_pedido',
            markers=True,
            title=f'Evolução do Número de Pedidos ({periodo_selecionado})',
            labels={'numero_pedido': 'Número de Pedidos', x_axis: 'Período'}
        )
        st.plotly_chart(fig_pedidos, use_container_width=True)
    
    with tab3:
        st.header("Análise por Categoria")
        
        # Seletor de categoria
        categorias = sorted([str(cat) for cat in df['categoria'].unique()])
        categoria_selecionada = st.selectbox("Selecione uma categoria para análise detalhada:", categorias)
        
        # Filtrar dados pela categoria selecionada
        df_cat = df[df['categoria'] == categoria_selecionada]
        
        # Métricas da categoria
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cat_vendas = df_cat['valor_total'].sum()
            percentual_vendas = (cat_vendas / df['valor_total'].sum()) * 100
            st.metric(
                "Total de Vendas", 
                f"R$ {cat_vendas:,.2f}",
                f"{percentual_vendas:.1f}% do total"
            )
        
        with col2:
            cat_produtos = df_cat['quantidade'].sum()
            percentual_produtos = (cat_produtos / df['quantidade'].sum()) * 100
            st.metric(
                "Produtos Vendidos", 
                f"{cat_produtos:,}",
                f"{percentual_produtos:.1f}% do total"
            )
        
        with col3:
            cat_pedidos = df_cat['numero_pedido'].nunique()
            percentual_pedidos = (cat_pedidos / df['numero_pedido'].nunique()) * 100
            st.metric(
                "Número de Pedidos", 
                f"{cat_pedidos:,}",
                f"{percentual_pedidos:.1f}% do total"
            )
        
        # Evolução temporal da categoria
        st.subheader(f"Evolução de Vendas - {categoria_selecionada}")
        df_cat['mes_ano'] = df_cat['data_venda'].dt.strftime('%Y-%m')
        df_cat_tempo = df_cat.groupby('mes_ano').agg({
            'valor_total': 'sum',
            'quantidade': 'sum'
        }).reset_index()
        
        fig_cat_tempo = px.line(
            df_cat_tempo,
            x='mes_ano',
            y='valor_total',
            markers=True,
            title=f'Evolução do Valor Total de Vendas - {categoria_selecionada}',
            labels={'valor_total': 'Valor Total (R$)', 'mes_ano': 'Mês/Ano'}
        )
        st.plotly_chart(fig_cat_tempo, use_container_width=True)
        
        # Comparação com outras categorias
        st.subheader("Comparação com Outras Categorias")
        df_comp = df.groupby('categoria').agg({
            'valor_total': 'sum',
            'quantidade': 'sum'
        }).reset_index()
        
        df_comp = df_comp.sort_values('valor_total', ascending=False)
        
        fig_comp = px.bar(
            df_comp,
            x='categoria',
            y='valor_total',
            title='Comparação de Vendas entre Categorias',
            color='categoria',
            labels={'valor_total': 'Valor Total (R$)', 'categoria': 'Categoria'}
        )
        
        # Destacar a categoria selecionada
        for i, bar in enumerate(fig_comp.data):
            if bar.name == categoria_selecionada:
                fig_comp.data[i].marker.line.width = 3
                fig_comp.data[i].marker.line.color = 'black'
        
        st.plotly_chart(fig_comp, use_container_width=True)
    
    with tab4:
        st.header("Insights e Recomendações")
        
        # Gerar insights baseados nos dados
        insights = generate_insights(df)
        
        # Exibir insights
        for i, insight in enumerate(insights):
            st.subheader(f"Insight {i+1}: {insight['titulo']}")
            st.write(insight['descricao'])
            
            if 'grafico' in insight:
                st.plotly_chart(insight['grafico'], use_container_width=True)
            
            st.markdown("---")
        
        # Recomendações baseadas nos insights
        st.subheader("Recomendações")
        
        # Categoria com maior crescimento
        df['mes_ano'] = df['data_venda'].dt.strftime('%Y-%m')
        df_crescimento = df.pivot_table(
            index='categoria',
            columns='mes_ano',
            values='valor_total',
            aggfunc='sum'
        ).fillna(0)
        
        if len(df_crescimento.columns) >= 2:
            df_crescimento['variacao'] = df_crescimento[df_crescimento.columns[-1]] / df_crescimento[df_crescimento.columns[-2]] - 1
            categoria_crescimento = df_crescimento['variacao'].idxmax()
            taxa_crescimento = df_crescimento.loc[categoria_crescimento, 'variacao'] * 100
            
            if taxa_crescimento > 0:
                st.info(f"📈 A categoria **{categoria_crescimento}** apresentou o maior crescimento recente ({taxa_crescimento:.1f}%). Considere aumentar o investimento nesta categoria.")
        
        # Categoria com maior ticket médio
        df_ticket = df.groupby('categoria').agg({
            'valor_total': 'sum',
            'numero_pedido': 'nunique'
        })
        df_ticket['ticket_medio'] = df_ticket['valor_total'] / df_ticket['numero_pedido']
        categoria_ticket = df_ticket['ticket_medio'].idxmax()
        ticket_max = df_ticket.loc[categoria_ticket, 'ticket_medio']
        
        st.info(f"💰 A categoria **{categoria_ticket}** possui o maior ticket médio (R$ {ticket_max:.2f}). Considere estratégias para aumentar o cross-selling nesta categoria.")
        
        # Dias da semana com melhor desempenho
        df['dia_semana'] = df['data_venda'].dt.day_name()
        df_dia = df.groupby('dia_semana').agg({
            'valor_total': 'sum'
        }).reset_index()
        
        # Mapear nomes dos dias para português e ordenar corretamente
        dias_semana_pt = {
            'Monday': 'Segunda-feira',
            'Tuesday': 'Terça-feira',
            'Wednesday': 'Quarta-feira',
            'Thursday': 'Quinta-feira',
            'Friday': 'Sexta-feira',
            'Saturday': 'Sábado',
            'Sunday': 'Domingo'
        }
        
        ordem_dias = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
        
        df_dia['dia_semana_pt'] = df_dia['dia_semana'].map(dias_semana_pt)
        df_dia = df_dia.sort_values(by='valor_total', ascending=False)
        
        melhor_dia = df_dia.iloc[0]['dia_semana_pt']
        st.info(f"📅 **{melhor_dia}** é o dia com maior volume de vendas. Considere programar promoções e campanhas para este dia da semana.")

else:
    # Exibir instruções quando nenhum arquivo for carregado
    st.info("👆 Faça o upload de uma planilha para visualizar o dashboard ou use os dados de exemplo.")
    
    st.markdown("""
    ### Formato esperado da planilha:
    
    A planilha deve conter as seguintes colunas:
    - **Número**: Identificador único do pedido
    - **Data da venda**: Data em que a venda foi realizada
    - **Quantidade de produtos**: Quantidade de itens vendidos
    - **Valor total da venda**: Valor monetário total da venda
    - **Categoria do produto**: Categoria do produto vendido
    - **Descrição do produto** (opcional): Descrição do produto para categorização automática
    
    Formatos aceitos: Excel (.xlsx) ou CSV (.csv)
    
    ### Categorização Automática
    
    Se a planilha incluir a coluna **Descrição do produto**, o sistema irá:
    - Identificar produtos sem categoria ou classificados como "Outros"
    - Atribuir automaticamente categorias com base na descrição do produto
    - Usar regras de palavras-chave e análise de similaridade de texto
    
    Isso melhora significativamente a qualidade dos dados para análise.
    """)
    
    # Exemplo de dashboard com dados fictícios
    st.markdown("### Exemplo de Dashboard (com dados fictícios)")
    
    # Criar dados de exemplo
    st.image("https://via.placeholder.com/800x400.png?text=Exemplo+de+Dashboard", use_column_width=True)

# Rodapé
st.markdown("---")
st.markdown("Dashboard de Análise de Vendas | Desenvolvido com Streamlit") 