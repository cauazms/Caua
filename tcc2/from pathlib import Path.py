# ============================================================
# LIMPEZA, INTEGRAÇÃO E PREPARAÇÃO DA BASE DE DENGUE - 2024
# São José do Rio Preto - SP
#
# BASES:
# 1. baseTratada (1).xlsx       -> Base epidemiológica
# 2. dengueSintomas.xlsx        -> Base de sintomas
# 3. CNES.xlsx                  -> Cadastro dos estabelecimentos
#
# INTEGRAÇÃO:
# ID_UNIDADE -> CNES
# ============================================================


import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. CONFIGURAÇÃO DOS ARQUIVOS
# ============================================================

PASTA = Path(r"C:\Users\Pichau\Downloads")

ARQUIVO_BASE = PASTA / "baseTratada (1).xlsx"
ARQUIVO_SINTOMAS = PASTA / "dengueSintomas.xlsx"

# Coloque aqui o nome exato da sua base CNES.
ARQUIVO_CNES = PASTA / "CNES.xlsx"

ARQUIVO_FINAL = PASTA / "DENGUE_2024_BASE_FINAL.xlsx"


# ============================================================
# 2. VERIFICAR OS ARQUIVOS
# ============================================================

print("=" * 70)
print("LIMPEZA E INTEGRAÇÃO DA BASE DE DENGUE")
print("São José do Rio Preto - SP | 2024")
print("=" * 70)


if not ARQUIVO_BASE.exists():
    raise FileNotFoundError(
        f"\nArquivo da base epidemiológica não encontrado:\n{ARQUIVO_BASE}"
    )


if not ARQUIVO_SINTOMAS.exists():
    raise FileNotFoundError(
        f"\nArquivo da base de sintomas não encontrado:\n{ARQUIVO_SINTOMAS}"
    )


if not ARQUIVO_CNES.exists():
    raise FileNotFoundError(
        f"""
Arquivo da base CNES não encontrado:

{ARQUIVO_CNES}

Verifique o nome do arquivo nesta linha:

ARQUIVO_CNES = PASTA / "CNES.xlsx"
"""
    )


# ============================================================
# 3. CARREGAR AS BASES
# ============================================================

print("\n[1/10] Carregando base epidemiológica...")

base = pd.read_excel(
    ARQUIVO_BASE
)

print("Base epidemiológica carregada.")


print("\n[2/10] Carregando base de sintomas...")

sintomas = pd.read_excel(
    ARQUIVO_SINTOMAS
)

print("Base de sintomas carregada.")


print("\n[3/10] Carregando base CNES...")

cnes = pd.read_excel(
    ARQUIVO_CNES,
    dtype={"CNES": "string"}
)

print("Base CNES carregada.")


print("\n" + "-" * 70)
print("INFORMAÇÕES INICIAIS")
print("-" * 70)

print(f"\nBase epidemiológica:")
print(f"Linhas: {len(base):,}")
print(f"Colunas: {len(base.columns)}")

print(f"\nBase de sintomas:")
print(f"Linhas: {len(sintomas):,}")
print(f"Colunas: {len(sintomas.columns)}")

print(f"\nBase CNES:")
print(f"Linhas: {len(cnes):,}")
print(f"Colunas: {len(cnes.columns)}")


# ============================================================
# 4. PADRONIZAR NOMES DAS COLUNAS
# ============================================================

def padronizar_colunas(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
    )

    return df


base = padronizar_colunas(base)
sintomas = padronizar_colunas(sintomas)
cnes = padronizar_colunas(cnes)


print("\nNomes das colunas padronizados.")


# ============================================================
# 5. VALIDAR COLUNAS DO CNES
# ============================================================

print("\n" + "-" * 70)
print("VALIDAÇÃO DA BASE CNES")
print("-" * 70)


if "CNES" not in cnes.columns:

    raise ValueError(
        """
A base CNES não possui a coluna 'CNES'.

Colunas encontradas:
""" + str(list(cnes.columns))
    )


if "ESTABELECIMENTO" not in cnes.columns:

    raise ValueError(
        """
A base CNES não possui a coluna 'ESTABELECIMENTO'.

Colunas encontradas:
""" + str(list(cnes.columns))
    )


if "ID_UNIDADE" not in base.columns:

    raise ValueError(
        """
A base epidemiológica não possui a coluna 'ID_UNIDADE'.

Colunas encontradas:
""" + str(list(base.columns))
    )


print("Coluna CNES encontrada.")
print("Coluna ESTABELECIMENTO encontrada.")
print("Coluna ID_UNIDADE encontrada.")


# ============================================================
# 6. VERIFICAR QUANTIDADE DA BASE DENGUE + SINTOMAS
# ============================================================

print("\n" + "-" * 70)
print("VALIDAÇÃO DENGUE + SINTOMAS")
print("-" * 70)


if len(base) != len(sintomas):

    raise ValueError(
        f"""
As bases epidemiológica e de sintomas possuem quantidades
diferentes de registros.

Base epidemiológica: {len(base):,}
Base sintomas:       {len(sintomas):,}

Não é seguro realizar a associação pela posição das linhas.
"""
    )


print(
    f"As duas bases possuem {len(base):,} registros."
)


# ============================================================
# 7. CRIAR IDENTIFICADOR DO REGISTRO
# ============================================================

base.insert(
    0,
    "ID_REGISTRO",
    range(1, len(base) + 1)
)


print("\nID_REGISTRO criado.")


# ============================================================
# 8. UNIR BASE EPIDEMIOLÓGICA + SINTOMAS
# ============================================================

print("\nIntegrando base epidemiológica + sintomas...")


base_final = pd.concat(
    [
        base.reset_index(drop=True),
        sintomas.reset_index(drop=True)
    ],
    axis=1
)


# Remove colunas duplicadas
base_final = base_final.loc[
    :,
    ~base_final.columns.duplicated()
]


print("\nBases epidemiológica e de sintomas integradas.")

print(
    f"Registros: {len(base_final):,}"
)

print(
    f"Colunas: {len(base_final.columns)}"
)


# ============================================================
# 9. CORRIGIR DATA DE NOTIFICAÇÃO
# ============================================================

if "DT_NOTIFIC" in base_final.columns:

    base_final["DT_NOTIFIC"] = pd.to_datetime(
        base_final["DT_NOTIFIC"],
        errors="coerce"
    )

    print("\nData de notificação convertida.")


# ============================================================
# 10. VALIDAR O ANO
# ============================================================

if "NU_ANO" in base_final.columns:

    base_final["NU_ANO"] = pd.to_numeric(
        base_final["NU_ANO"],
        errors="coerce"
    )

    antes_ano = len(base_final)

    base_final = base_final[
        base_final["NU_ANO"] == 2024
    ].copy()

    depois_ano = len(base_final)

    print("\nValidação do ano:")
    print(f"Antes:  {antes_ano:,}")
    print(f"Depois: {depois_ano:,}")


# ============================================================
# 11. VALIDAR MUNICÍPIO
# ============================================================

CODIGO_MUNICIPIO = 354980


if "ID_MUNICIP" in base_final.columns:

    base_final["ID_MUNICIP"] = pd.to_numeric(
        base_final["ID_MUNICIP"],
        errors="coerce"
    )

    antes_municipio = len(base_final)

    base_final = base_final[
        base_final["ID_MUNICIP"] == CODIGO_MUNICIPIO
    ].copy()

    depois_municipio = len(base_final)

    print("\nValidação do município:")
    print(f"Antes:  {antes_municipio:,}")
    print(f"Depois: {depois_municipio:,}")


# ============================================================
# 12. PADRONIZAR SEXO
# ============================================================

if "CS_SEXO" in base_final.columns:

    base_final["CS_SEXO"] = (
        base_final["CS_SEXO"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    mapa_sexo = {

        "F": "Feminino",

        "M": "Masculino",

        "I": "Ignorado"
    }

    base_final["SEXO"] = (
        base_final["CS_SEXO"]
        .map(mapa_sexo)
        .fillna("Não informado")
    )

    print("\nSexo padronizado.")


# ============================================================
# 13. CORRIGIR ANO DE NASCIMENTO
# ============================================================

if "ANO_NASC" in base_final.columns:

    base_final["ANO_NASC"] = pd.to_numeric(
        base_final["ANO_NASC"],
        errors="coerce"
    )

    base_final.loc[
        (base_final["ANO_NASC"] < 1900) |
        (base_final["ANO_NASC"] > 2024),
        "ANO_NASC"
    ] = np.nan

    print("\nAno de nascimento validado.")


# ============================================================
# 14. CRIAR IDADE
# ============================================================

if "ANO_NASC" in base_final.columns:

    base_final["IDADE"] = (
        base_final["NU_ANO"] -
        base_final["ANO_NASC"]
    )

    base_final.loc[
        (base_final["IDADE"] < 0) |
        (base_final["IDADE"] > 120),
        "IDADE"
    ] = np.nan

    print("Idade calculada.")


# ============================================================
# 15. CRIAR FAIXA ETÁRIA
# ============================================================

if "IDADE" in base_final.columns:

    bins = [
        -1,
        4,
        9,
        14,
        19,
        29,
        39,
        49,
        59,
        69,
        79,
        120
    ]

    labels = [
        "0-4",
        "5-9",
        "10-14",
        "15-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70-79",
        "80+"
    ]

    base_final["FAIXA_ETARIA"] = pd.cut(
        base_final["IDADE"],
        bins=bins,
        labels=labels
    )

    print("Faixa etária criada.")


# ============================================================
# 16. PADRONIZAR SINTOMAS
# ============================================================

COLUNAS_SINTOMAS = [

    "MIALGIA",

    "CEFALEIA",

    "EXANTEMA",

    "VOMITO",

    "NAUSEA",

    "DOR_COSTAS",

    "CONJUNTVIT",

    "ARTRITE",

    "ARTRALGIA",

    "DOR_RETRO"
]


print("\nPadronizando sintomas...")


for coluna in COLUNAS_SINTOMAS:

    if coluna in base_final.columns:

        base_final[coluna] = pd.to_numeric(
            base_final[coluna],
            errors="coerce"
        )

        nova_coluna = coluna + "_DESC"

        base_final[nova_coluna] = (

            base_final[coluna]

            .map({

                1: "Sim",

                2: "Não",

                9: "Ignorado"
            })

            .fillna("Não informado")
        )


print("Sintomas padronizados.")


# ============================================================
# 17. QUANTIDADE DE SINTOMAS
# ============================================================

colunas_existentes = [

    c

    for c in COLUNAS_SINTOMAS

    if c in base_final.columns
]


if colunas_existentes:

    base_final["QTD_SINTOMAS"] = (

        base_final[
            colunas_existentes
        ]

        .eq(1)

        .sum(axis=1)
    )

    print("Quantidade de sintomas calculada.")


# ============================================================
# 18. CLASSIFICAÇÃO DOS SINTOMAS
# ============================================================

if "QTD_SINTOMAS" in base_final.columns:

    base_final["CLASSIFICACAO_SINTOMAS"] = pd.cut(

        base_final["QTD_SINTOMAS"],

        bins=[
            -1,
            0,
            2,
            5,
            10
        ],

        labels=[
            "Nenhum",
            "1-2 sintomas",
            "3-5 sintomas",
            "6+ sintomas"
        ]
    )


# ============================================================
# 19. MÊS DA NOTIFICAÇÃO
# ============================================================

if "DT_NOTIFIC" in base_final.columns:

    base_final["MES"] = (
        base_final["DT_NOTIFIC"]
        .dt.month
    )

    mapa_meses = {

        1: "Janeiro",

        2: "Fevereiro",

        3: "Março",

        4: "Abril",

        5: "Maio",

        6: "Junho",

        7: "Julho",

        8: "Agosto",

        9: "Setembro",

        10: "Outubro",

        11: "Novembro",

        12: "Dezembro"
    }

    base_final["MES_NOME"] = (

        base_final["MES"]

        .map(mapa_meses)
    )

    print("Mês da notificação criado.")


# ============================================================
# 20. TRIMESTRE
# ============================================================

if "DT_NOTIFIC" in base_final.columns:

    base_final["TRIMESTRE"] = (

        base_final["DT_NOTIFIC"]

        .dt.quarter

        .map({

            1: "1º Trimestre",

            2: "2º Trimestre",

            3: "3º Trimestre",

            4: "4º Trimestre"
        })
    )


# ============================================================
# 21. PREPARAR ID_UNIDADE PARA INTEGRAÇÃO COM CNES
# ============================================================

print("\n" + "-" * 70)
print("INTEGRAÇÃO COM CNES")
print("-" * 70)


def padronizar_codigo_cnes(valor):

    if pd.isna(valor):

        return pd.NA

    texto = str(valor).strip()

    # Caso o Excel tenha convertido 1234567.0
    if texto.endswith(".0"):

        texto = texto[:-2]

    # Remove espaços
    texto = texto.strip()

    # Se for somente número, garante 7 dígitos
    if texto.isdigit():

        texto = texto.zfill(7)

    return texto


# ============================================================
# 22. PADRONIZAR ID_UNIDADE
# ============================================================

base_final["ID_UNIDADE_CNES"] = (

    base_final["ID_UNIDADE"]

    .apply(padronizar_codigo_cnes)

    .astype("string")
)


# ============================================================
# 23. PADRONIZAR CNES
# ============================================================

cnes["CNES"] = (

    cnes["CNES"]

    .apply(padronizar_codigo_cnes)

    .astype("string")
)


print("ID_UNIDADE padronizado.")
print("CNES padronizado.")


# ============================================================
# 24. VERIFICAR DUPLICIDADE NO CNES
# ============================================================

duplicados_cnes = (

    cnes["CNES"]

    .duplicated()

    .sum()
)


print(
    f"\nCNES duplicados encontrados: "
    f"{duplicados_cnes:,}"
)


# ============================================================
# 25. PREPARAR BASE CNES
# ============================================================

colunas_cnes = [

    "CNES",

    "ESTABELECIMENTO"
]


# Adicionar CNPJ se existir
if "CNPJ" in cnes.columns:

    colunas_cnes.append("CNPJ")


# Adicionar GESTÃO se existir
if "GESTÃO" in cnes.columns:

    colunas_cnes.append("GESTÃO")


cnes_integracao = (

    cnes[colunas_cnes]

    .drop_duplicates(
        subset=["CNES"],
        keep="first"
    )

    .copy()
)


# ============================================================
# 26. REALIZAR MERGE COM CNES
# ============================================================

print("\nRealizando integração:")

print(
    "ID_UNIDADE -> CNES"
)


base_final = base_final.merge(

    cnes_integracao,

    left_on="ID_UNIDADE_CNES",

    right_on="CNES",

    how="left",

    suffixes=("", "_CNES")
)


# ============================================================
# 27. VERIFICAR CORRESPONDÊNCIAS CNES
# ============================================================

total_registros = len(base_final)

registros_com_cnes = (

    base_final["CNES"]

    .notna()

    .sum()
)


registros_sem_cnes = (

    base_final["CNES"]

    .isna()

    .sum()
)


percentual_cnes = (

    registros_com_cnes /

    total_registros *

    100

    if total_registros > 0

    else 0
)


print("\nResultado da integração CNES:")

print(
    f"Total de registros: "
    f"{total_registros:,}"
)

print(
    f"Com correspondência CNES: "
    f"{registros_com_cnes:,}"
)

print(
    f"Sem correspondência CNES: "
    f"{registros_sem_cnes:,}"
)

print(
    f"Percentual integrado: "
    f"{percentual_cnes:.2f}%"
)


# ============================================================
# 28. CRIAR INDICADOR DE INTEGRAÇÃO
# ============================================================

base_final["STATUS_CNES"] = np.where(

    base_final["CNES"].notna(),

    "Encontrado no CNES",

    "Não encontrado no CNES"
)


# ============================================================
# 29. REMOVER COLUNA AUXILIAR
# ============================================================

base_final.drop(

    columns=["ID_UNIDADE_CNES"],

    inplace=True,

    errors="ignore"
)


# ============================================================
# 30. TRATAR VALORES AUSENTES
# ============================================================

print("\nTratando valores ausentes...")


for coluna in base_final.columns:

    if base_final[coluna].dtype == "object":

        base_final[coluna] = (

            base_final[coluna]

            .fillna("Não informado")
        )


print("Valores ausentes tratados.")


# ============================================================
# 31. VERIFICAR DUPLICIDADES DO ID_REGISTRO
# ============================================================

duplicados = (

    base_final

    .duplicated(
        subset=["ID_REGISTRO"]
    )

    .sum()
)


print("\nDuplicidades no ID_REGISTRO:")

print(duplicados)


# Não removemos registros apenas porque
# os demais campos são iguais.


# ============================================================
# 32. RELATÓRIO DE QUALIDADE
# ============================================================

relatorio = pd.DataFrame({

    "COLUNA": base_final.columns,

    "TIPO": [

        str(base_final[c].dtype)

        for c in base_final.columns
    ],

    "NULOS": [

        base_final[c].isna().sum()

        for c in base_final.columns
    ],

    "PERCENTUAL_NULOS": [

        round(
            base_final[c].isna().mean() * 100,
            2
        )

        for c in base_final.columns
    ],

    "VALORES_UNICOS": [

        base_final[c].nunique()

        for c in base_final.columns
    ]
})


# ============================================================
# 33. RESUMO GERAL
# ============================================================

resumo = pd.DataFrame({

    "INDICADOR": [

        "Total de registros",

        "Total de colunas",

        "Ano",

        "Município",

        "Código IBGE",

        "Registros encontrados no CNES",

        "Registros não encontrados no CNES",

        "Percentual integrado ao CNES"
    ],

    "VALOR": [

        len(base_final),

        len(base_final.columns),

        2024,

        "São José do Rio Preto",

        CODIGO_MUNICIPIO,

        registros_com_cnes,

        registros_sem_cnes,

        round(percentual_cnes, 2)
    ]
})


# ============================================================
# 34. RESUMO DOS SINTOMAS
# ============================================================

resumo_sintomas = []


for coluna in COLUNAS_SINTOMAS:

    if coluna in base_final.columns:

        quantidade = (

            base_final[coluna] == 1

        ).sum()


        percentual = (

            quantidade /

            len(base_final) *

            100

            if len(base_final) > 0

            else 0
        )


        resumo_sintomas.append({

            "SINTOMA": coluna,

            "CASOS": quantidade,

            "PERCENTUAL": round(
                percentual,
                2
            )
        })


resumo_sintomas = pd.DataFrame(
    resumo_sintomas
)


if not resumo_sintomas.empty:

    resumo_sintomas = (

        resumo_sintomas

        .sort_values(
            "CASOS",
            ascending=False
        )
    )


# ============================================================
# 35. RESUMO POR SEXO
# ============================================================

if "SEXO" in base_final.columns:

    resumo_sexo = (

        base_final["SEXO"]

        .value_counts()

        .reset_index()
    )

    resumo_sexo.columns = [

        "SEXO",

        "QUANTIDADE"
    ]

else:

    resumo_sexo = pd.DataFrame()


# ============================================================
# 36. RESUMO POR FAIXA ETÁRIA
# ============================================================

if "FAIXA_ETARIA" in base_final.columns:

    resumo_idade = (

        base_final["FAIXA_ETARIA"]

        .value_counts()

        .sort_index()

        .reset_index()
    )

    resumo_idade.columns = [

        "FAIXA_ETARIA",

        "QUANTIDADE"
    ]

else:

    resumo_idade = pd.DataFrame()


# ============================================================
# 37. RESUMO POR MÊS
# ============================================================

if "MES_NOME" in base_final.columns:

    resumo_mes = (

        base_final

        .groupby(
            ["MES", "MES_NOME"]
        )

        .size()

        .reset_index(
            name="CASOS"
        )

        .sort_values("MES")
    )

else:

    resumo_mes = pd.DataFrame()


# ============================================================
# 38. RESUMO POR ESTABELECIMENTO
# ============================================================

if "ESTABELECIMENTO" in base_final.columns:

    resumo_estabelecimento = (

        base_final

        .groupby(
            "ESTABELECIMENTO"
        )

        .size()

        .reset_index(
            name="CASOS"
        )

        .sort_values(
            "CASOS",
            ascending=False
        )
    )

else:

    resumo_estabelecimento = pd.DataFrame()


# ============================================================
# 39. RESUMO DA INTEGRAÇÃO CNES
# ============================================================

resumo_cnes = pd.DataFrame({

    "STATUS": [

        "Encontrado no CNES",

        "Não encontrado no CNES"
    ],

    "QUANTIDADE": [

        registros_com_cnes,

        registros_sem_cnes
    ],

    "PERCENTUAL": [

        round(
            registros_com_cnes /
            total_registros *
            100,
            2
        )
        if total_registros > 0
        else 0,

        round(
            registros_sem_cnes /
            total_registros *
            100,
            2
        )
        if total_registros > 0
        else 0
    ]
})


# ============================================================
# 40. SALVAR TUDO EM UM ÚNICO EXCEL
# ============================================================

print("\n" + "-" * 70)

print("SALVANDO ARQUIVO FINAL...")

print("-" * 70)


with pd.ExcelWriter(

    ARQUIVO_FINAL,

    engine="openpyxl"

) as writer:


    # --------------------------------------------------------
    # BASE FINAL
    # --------------------------------------------------------

    base_final.to_excel(

        writer,

        sheet_name="BASE_FINAL",

        index=False
    )


    # --------------------------------------------------------
    # QUALIDADE
    # --------------------------------------------------------

    relatorio.to_excel(

        writer,

        sheet_name="QUALIDADE",

        index=False
    )


    # --------------------------------------------------------
    # RESUMO
    # --------------------------------------------------------

    resumo.to_excel(

        writer,

        sheet_name="RESUMO",

        index=False
    )


    # --------------------------------------------------------
    # SINTOMAS
    # --------------------------------------------------------

    if not resumo_sintomas.empty:

        resumo_sintomas.to_excel(

            writer,

            sheet_name="SINTOMAS",

            index=False
        )


    # --------------------------------------------------------
    # SEXO
    # --------------------------------------------------------

    if not resumo_sexo.empty:

        resumo_sexo.to_excel(

            writer,

            sheet_name="SEXO",

            index=False
        )


    # --------------------------------------------------------
    # FAIXA ETÁRIA
    # --------------------------------------------------------

    if not resumo_idade.empty:

        resumo_idade.to_excel(

            writer,

            sheet_name="FAIXA_ETARIA",

            index=False
        )


    # --------------------------------------------------------
    # MÊS
    # --------------------------------------------------------

    if not resumo_mes.empty:

        resumo_mes.to_excel(

            writer,

            sheet_name="MES",

            index=False
        )


    # --------------------------------------------------------
    # ESTABELECIMENTO
    # --------------------------------------------------------

    if not resumo_estabelecimento.empty:

        resumo_estabelecimento.to_excel(

            writer,

            sheet_name="ESTABELECIMENTOS",

            index=False
        )


    # --------------------------------------------------------
    # CNES
    # --------------------------------------------------------

    resumo_cnes.to_excel(

        writer,

        sheet_name="INTEGRACAO_CNES",

        index=False
    )


# ============================================================
# 41. RESULTADO FINAL
# ============================================================

print("\n")

print("=" * 70)

print("LIMPEZA E INTEGRAÇÃO CONCLUÍDAS COM SUCESSO!")

print("=" * 70)


print(
    f"\nTotal final de registros: "
    f"{len(base_final):,}"
)


print(
    f"Total de colunas: "
    f"{len(base_final.columns)}"
)


print(
    f"\nRegistros integrados ao CNES: "
    f"{registros_com_cnes:,}"
)


print(
    f"Registros sem correspondência: "
    f"{registros_sem_cnes:,}"
)


print(
    f"Percentual de integração: "
    f"{percentual_cnes:.2f}%"
)


print("\nArquivo criado:")

print(ARQUIVO_FINAL)


print("\nPlanilhas criadas:")

print("1. BASE_FINAL")

print("2. QUALIDADE")

print("3. RESUMO")

print("4. SINTOMAS")

print("5. SEXO")

print("6. FAIXA_ETARIA")

print("7. MES")

print("8. ESTABELECIMENTOS")

print("9. INTEGRACAO_CNES")


print("\nProcesso finalizado.")