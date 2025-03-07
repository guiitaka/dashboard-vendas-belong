import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def process_data(df):
    """
    Processa os dados para análise.
    
    Args:
        df (DataFrame): DataFrame com os dados brutos
        
    Returns:
        DataFrame: DataFrame processado
    """
    # Cópia para não modificar o original
    df_processed = df.copy()
    
    # Mapear os nomes das colunas da planilha para os nomes usados no código
    column_mapping = {
        'Número': 'numero_pedido',
        'Data da venda': 'data_venda',
        'Quantidade de produtos': 'quantidade',
        'Valor total da venda': 'valor_total',
        'Categoria do produto': 'categoria'  # Atualizado para o nome correto
    }
    
    # Renomear as colunas
    for original, novo in column_mapping.items():
        if original in df_processed.columns:
            df_processed = df_processed.rename(columns={original: novo})
    
    # Garantir que as colunas necessárias existam
    required_columns = ['data_venda', 'quantidade', 'valor_total', 'categoria']
    for col in required_columns:
        if col not in df_processed.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no DataFrame")
    
    # Converter data para datetime se ainda não estiver
    if not pd.api.types.is_datetime64_any_dtype(df_processed['data_venda']):
        df_processed['data_venda'] = pd.to_datetime(df_processed['data_venda'])
    
    # Adicionar colunas úteis para análise
    df_processed['mes'] = df_processed['data_venda'].dt.month
    df_processed['ano'] = df_processed['data_venda'].dt.year
    df_processed['dia_semana'] = df_processed['data_venda'].dt.dayofweek
    df_processed['semana_ano'] = df_processed['data_venda'].dt.isocalendar().week
    
    # Calcular métricas adicionais
    if 'numero_pedido' in df_processed.columns:
        # Calcular valor médio por pedido para cada categoria
        df_processed['valor_medio_pedido'] = df_processed.groupby('numero_pedido')['valor_total'].transform('mean')
    
    # Calcular valor médio por produto
    df_processed['valor_medio_produto'] = df_processed['valor_total'] / df_processed['quantidade']
    
    return df_processed

def generate_insights(df):
    """
    Gera insights baseados nos dados.
    
    Args:
        df (DataFrame): DataFrame com os dados processados
        
    Returns:
        list: Lista de dicionários com insights
    """
    insights = []
    
    # Insight 1: Categoria mais vendida
    categoria_mais_vendida = df.groupby('categoria')['valor_total'].sum().idxmax()
    valor_categoria = df[df['categoria'] == categoria_mais_vendida]['valor_total'].sum()
    percentual = (valor_categoria / df['valor_total'].sum()) * 100
    
    # Gráfico para o insight 1
    df_cat_vendas = df.groupby('categoria')['valor_total'].sum().reset_index()
    df_cat_vendas = df_cat_vendas.sort_values('valor_total', ascending=False)
    
    fig_cat_vendas = px.bar(
        df_cat_vendas,
        x='categoria',
        y='valor_total',
        title='Valor Total de Vendas por Categoria',
        color='categoria',
        labels={'valor_total': 'Valor Total (R$)', 'categoria': 'Categoria'}
    )
    
    insights.append({
        'titulo': f"A categoria mais vendida é '{categoria_mais_vendida}'",
        'descricao': f"A categoria '{categoria_mais_vendida}' representa {percentual:.1f}% do valor total de vendas, totalizando R$ {valor_categoria:,.2f}.",
        'grafico': fig_cat_vendas
    })
    
    # Insight 2: Tendência de crescimento
    if 'data_venda' in df.columns:
        df['mes_ano'] = df['data_venda'].dt.strftime('%Y-%m')
        df_tendencia = df.groupby('mes_ano')['valor_total'].sum().reset_index()
        
        if len(df_tendencia) > 1:
            primeiro_mes = df_tendencia.iloc[0]['valor_total']
            ultimo_mes = df_tendencia.iloc[-1]['valor_total']
            variacao = ((ultimo_mes / primeiro_mes) - 1) * 100
            
            fig_tendencia = px.line(
                df_tendencia,
                x='mes_ano',
                y='valor_total',
                markers=True,
                title='Tendência de Vendas ao Longo do Tempo',
                labels={'valor_total': 'Valor Total (R$)', 'mes_ano': 'Mês/Ano'}
            )
            
            status = "crescimento" if variacao > 0 else "queda"
            
            insights.append({
                'titulo': f"Tendência de {status} nas vendas",
                'descricao': f"As vendas apresentaram {status} de {abs(variacao):.1f}% comparando o primeiro e o último período analisados.",
                'grafico': fig_tendencia
            })
    
    # Insight 3: Sazonalidade semanal
    if 'data_venda' in df.columns:
        # Mapear dias da semana para português
        dias_semana = {
            0: 'Segunda-feira',
            1: 'Terça-feira',
            2: 'Quarta-feira',
            3: 'Quinta-feira',
            4: 'Sexta-feira',
            5: 'Sábado',
            6: 'Domingo'
        }
        
        df['dia_semana_nome'] = df['data_venda'].dt.dayofweek.map(dias_semana)
        df_dia_semana = df.groupby('dia_semana_nome')['valor_total'].sum().reset_index()
        
        # Ordenar os dias da semana corretamente
        ordem_dias = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
        df_dia_semana['ordem'] = df_dia_semana['dia_semana_nome'].map({dia: i for i, dia in enumerate(ordem_dias)})
        df_dia_semana = df_dia_semana.sort_values('ordem')
        df_dia_semana = df_dia_semana.drop('ordem', axis=1)
        
        melhor_dia = df_dia_semana.loc[df_dia_semana['valor_total'].idxmax(), 'dia_semana_nome']
        pior_dia = df_dia_semana.loc[df_dia_semana['valor_total'].idxmin(), 'dia_semana_nome']
        
        fig_dia_semana = px.bar(
            df_dia_semana,
            x='dia_semana_nome',
            y='valor_total',
            title='Vendas por Dia da Semana',
            labels={'valor_total': 'Valor Total (R$)', 'dia_semana_nome': 'Dia da Semana'},
            color='dia_semana_nome'
        )
        
        insights.append({
            'titulo': "Sazonalidade semanal nas vendas",
            'descricao': f"O melhor dia para vendas é {melhor_dia}, enquanto {pior_dia} apresenta o menor volume de vendas.",
            'grafico': fig_dia_semana
        })
    
    # Insight 4: Relação entre quantidade e valor
    if len(df) > 0:
        correlacao = df[['quantidade', 'valor_total']].corr().iloc[0, 1]
        
        fig_scatter = px.scatter(
            df,
            x='quantidade',
            y='valor_total',
            color='categoria',
            title='Relação entre Quantidade de Produtos e Valor Total',
            labels={'quantidade': 'Quantidade de Produtos', 'valor_total': 'Valor Total (R$)'},
            trendline='ols'
        )
        
        if correlacao > 0.7:
            relacao = "forte correlação positiva"
        elif correlacao > 0.3:
            relacao = "correlação positiva moderada"
        elif correlacao > -0.3:
            relacao = "correlação fraca"
        else:
            relacao = "correlação negativa"
        
        insights.append({
            'titulo': f"Relação entre quantidade e valor total",
            'descricao': f"Existe uma {relacao} (coeficiente: {correlacao:.2f}) entre a quantidade de produtos e o valor total das vendas.",
            'grafico': fig_scatter
        })
    
    # Insight 5: Categorias com maior ticket médio
    if 'numero_pedido' in df.columns:
        df_ticket = df.groupby(['categoria', 'numero_pedido'])['valor_total'].sum().reset_index()
        df_ticket_medio = df_ticket.groupby('categoria')['valor_total'].mean().reset_index()
        df_ticket_medio = df_ticket_medio.sort_values('valor_total', ascending=False)
        df_ticket_medio = df_ticket_medio.rename(columns={'valor_total': 'ticket_medio'})
        
        categoria_maior_ticket = df_ticket_medio.iloc[0]['categoria']
        valor_maior_ticket = df_ticket_medio.iloc[0]['ticket_medio']
        
        fig_ticket = px.bar(
            df_ticket_medio,
            x='categoria',
            y='ticket_medio',
            title='Ticket Médio por Categoria',
            labels={'ticket_medio': 'Ticket Médio (R$)', 'categoria': 'Categoria'},
            color='categoria'
        )
        
        insights.append({
            'titulo': f"Categoria com maior ticket médio: {categoria_maior_ticket}",
            'descricao': f"A categoria '{categoria_maior_ticket}' possui o maior ticket médio (R$ {valor_maior_ticket:.2f}), o que indica potencial para estratégias de upselling.",
            'grafico': fig_ticket
        })
    
    return insights 