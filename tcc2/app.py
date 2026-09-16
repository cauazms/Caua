import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import requests
import re
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
import folium
from streamlit_folium import st_folium

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(
    page_title="Dashboard Dengue 2024 - São José do Rio Preto",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_PADRAO = BASE_DIR / "DENGUE_2024_BASE_COMPLETA_COM_CNES (1).xlsx"
ARQUIVO_COORDENADAS = BASE_DIR / "coordenadas_cnes.csv"

# ============================================================
# GEOLOCALIZAÇÃO
# ============================================================
# O mapa NÃO depende mais de pesquisar "CNES XXXXXXX" no OSM.
# Primeiro usamos o próprio cadastro do CNES para obter o endereço
# oficial da unidade. Depois o endereço é convertido em coordenadas
# por um geocodificador. O Nominatim fica apenas como último fallback.

CNES_IBGE_MUNICIPIO = "354980"  # São José do Rio Preto
CNES_FICHA_URL = "https://cnes2.datasus.gov.br/Mod_Conjunto.asp?VCo_Unidade={ibge}{cnes}"

geolocator = Nominatim(user_agent="dashboard_dengue_sao_jose_rio_preto_osm")
geocode_nominatim = RateLimiter(geolocator.geocode, min_delay_seconds=1)

SESSION_CNES = requests.Session()
SESSION_CNES.headers.update({
    "User-Agent": "Mozilla/5.0 (Dashboard Dengue 2024 - CNES/DATASUS)"
})

SINTOMAS = {
    "MIALGIA": "Mialgia",
    "CEFALEIA": "Cefaleia",
    "EXANTEMA": "Exantema",
    "VOMITO": "Vômito",
    "NAUSEA": "Náusea",
    "DOR_COSTAS": "Dor nas costas",
    "CONJUNTVIT": "Conjuntivite",
    "ARTRITE": "Artrite",
    "ARTRALGIA": "Artralgia",
    "DOR_RETRO": "Dor retro-orbitária",
}

# ============================================================
# FUNÇÕES
# ============================================================
@st.cache_data
def carregar_base(arquivo):
    df = pd.read_excel(arquivo, sheet_name="BASE_COMPLETA")

    # Datas
    df["DT_NOTIFIC"] = pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")

    # CNES / unidade como texto de 7 dígitos
    for col in ["ID_UNIDADE", "CNES"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.replace(r"\.0$", "", regex=True)
                .str.strip()
                .str.zfill(7)
            )

    # Ano de nascimento / idade
    df["ANO_NASC"] = pd.to_numeric(df["ANO_NASC"], errors="coerce")
    df["IDADE"] = df["DT_NOTIFIC"].dt.year - df["ANO_NASC"]
    df.loc[(df["IDADE"] < 0) | (df["IDADE"] > 120), "IDADE"] = np.nan

    def faixa(idade):
        if pd.isna(idade):
            return "Sem informação"
        idade = int(idade)
        if idade <= 9:
            return "0–9"
        if idade <= 19:
            return "10–19"
        if idade <= 29:
            return "20–29"
        if idade <= 39:
            return "30–39"
        if idade <= 49:
            return "40–49"
        if idade <= 59:
            return "50–59"
        return "60+"

    df["FAIXA_ETARIA"] = df["IDADE"].apply(faixa)

    # Calendário
    meses = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
        5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
        9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
    }
    df["MES_NUM"] = df["DT_NOTIFIC"].dt.month
    df["MES"] = df["MES_NUM"].map(meses)
    df["TRIMESTRE"] = df["DT_NOTIFIC"].dt.quarter.map(
        {1: "1º trimestre", 2: "2º trimestre", 3: "3º trimestre", 4: "4º trimestre"}
    )

    # Sexo
    df["SEXO_DESC"] = df["CS_SEXO"].map({
        "F": "Feminino",
        "M": "Masculino",
        "I": "Ignorado",
    }).fillna("Sem informação")

    # Sorotipo: mantemos os códigos presentes na base, sem inventar rótulos
    df["SOROTIPO_DESC"] = df["SOROTIPO"].apply(
        lambda x: f"Código {int(x)}" if pd.notna(x) else "Sem informação"
    )

    # Sintomas: no SINAN, 1 = sim e 2 = não para estas variáveis.
    # Valores diferentes de 1/2 ficam como desconhecidos.
    qtd = []
    for _, row in df.iterrows():
        n = 0
        for col in SINTOMAS:
            if pd.notna(row.get(col)) and float(row[col]) == 1:
                n += 1
        qtd.append(n)
    df["QTD_SINTOMAS"] = qtd

    def classe_sintomas(n):
        if n == 0:
            return "Nenhum registrado"
        if n == 1:
            return "1 sintoma"
        if n == 2:
            return "2 sintomas"
        if n <= 4:
            return "3–4 sintomas"
        return "5+ sintomas"

    df["CLASSIFICACAO_SINTOMAS"] = df["QTD_SINTOMAS"].apply(classe_sintomas)

    return df



@st.cache_data(show_spinner=False)
def carregar_cadastro_cnes(arquivo):
    """Carrega a aba BASE_CNES da própria planilha fornecida."""
    cnes = pd.read_excel(arquivo, sheet_name="BASE_CNES")
    colunas = [c for c in ["ESTABELECIMENTO", "CNES", "CNPJ", "GESTÃO"] if c in cnes.columns]
    cnes = cnes[colunas].copy()
    if "CNES" in cnes.columns:
        cnes["CNES"] = (cnes["CNES"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(7))
    if "ESTABELECIMENTO" in cnes.columns:
        cnes["ESTABELECIMENTO"] = cnes["ESTABELECIMENTO"].astype("string").str.strip()
    return cnes.drop_duplicates(subset=["CNES"])


def montar_endereco(row):
    """Monta endereço a partir dos campos disponíveis."""
    partes = []
    candidatos = [
        "LOGRADOURO", "ENDERECO", "ENDEREÇO", "NUMERO", "NÚMERO",
        "BAIRRO", "MUNICIPIO", "MUNICÍPIO", "UF", "CEP"
    ]
    for col in candidatos:
        if col in row.index and pd.notna(row[col]):
            valor = str(row[col]).strip()
            if valor and valor.lower() not in {"nan", "none", "-"}:
                partes.append(valor)

    estabelecimento = row.get("ESTABELECIMENTO")
    if pd.notna(estabelecimento) and str(estabelecimento).strip() not in {"", "nan", "None"}:
        partes.insert(0, str(estabelecimento).strip())

    if not partes:
        return None

    endereco = ", ".join(partes)
    if "rio preto" not in endereco.lower():
        endereco += ", São José do Rio Preto"
    if "sp" not in endereco.lower():
        endereco += ", SP"
    if "brasil" not in endereco.lower():
        endereco += ", Brasil"
    return endereco


def _normalizar_cnes(serie):
    return (
        serie.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(7)
    )


def _texto_limpo(valor):
    if valor is None:
        return ""
    return re.sub(r"\s+", " ", str(valor)).strip()


def consultar_cnes_oficial(cnes):
    """Consulta a ficha pública do CNES pelo número que já existe na planilha.

    Retorna endereço oficial e, se a página disponibilizar coordenadas,
    também tenta aproveitá-las diretamente.
    """
    cnes = _texto_limpo(cnes)
    if not cnes or cnes in {"0000000", "nan", "None", "<NA>"}:
        return None

    cnes = re.sub(r"\.0$", "", cnes).zfill(7)
    url = CNES_FICHA_URL.format(ibge=CNES_IBGE_MUNICIPIO, cnes=cnes)

    try:
        resposta = SESSION_CNES.get(url, timeout=20)
        resposta.raise_for_status()
        html = resposta.text
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            texto = _texto_limpo(soup.get_text(" ", strip=True))
        else:
            # Fallback sem BeautifulSoup: remove scripts, estilos e tags HTML.
            texto = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", " ", html, flags=re.I | re.S)
            texto = re.sub(r"<[^>]+>", " ", texto)
            texto = _texto_limpo(texto)

        if not texto or cnes not in texto:
            return None

        dados = {
            "CNES": cnes,
            "URL_CNES": url,
            "ENDERECO_CNES": None,
            "LATITUDE_CNES": np.nan,
            "LONGITUDE_CNES": np.nan,
            "STATUS_CNES_GEO": "Consulta CNES realizada",
        }

        # A ficha pública apresenta os campos nesta ordem.
        m = re.search(
            r"Logradouro:\s*N[úu]mero:\s*Telefone:\s*(.*?)\s*Complemento:\s*Bairro:\s*CEP:\s*Munic[ií]pio:\s*UF:\s*(.*?)\s*Tipo Estabelecimento:",
            texto,
            flags=re.IGNORECASE,
        )
        if m:
            logradouro_numero = _texto_limpo(m.group(1))
            # A ficha antiga coloca o telefone imediatamente depois do número.
            logradouro_numero = re.sub(
                r"\s*(?:\(?\d{2}\)?\s*)?\d{4,5}[- ]?\d{4}\s*$",
                "",
                logradouro_numero,
            ).strip()
            complemento_bairro_cep_municipio = _texto_limpo(m.group(2))

            # O segundo bloco costuma ser: complemento bairro CEP município - UF - IBGE - código UF
            cep_match = re.search(r"\b(\d{5}-?\d{3})\b", complemento_bairro_cep_municipio)
            cep = cep_match.group(1) if cep_match else ""

            municipio_match = re.search(
                r"(SAO JOSE DO RIO PRETO|SÃO JOSÉ DO RIO PRETO)\s*-\s*IBGE\s*-\s*\d+\s+([A-Z]{2})",
                complemento_bairro_cep_municipio,
                flags=re.IGNORECASE,
            )
            if municipio_match:
                municipio = "São José do Rio Preto"
                uf = municipio_match.group(1) if municipio_match.group(1).upper() in {"SP", "RJ", "MG", "PR", "MS", "GO", "BA", "DF", "SC", "RS", "ES", "PE", "CE", "PA", "AM", "MA", "PI", "RN", "PB", "AL", "SE", "MT", "RO", "AC", "AP", "RR", "TO"} else municipio_match.group(2)
                # Na ficha, o grupo capturado como UF é o segundo grupo.
                uf = municipio_match.group(2).upper()
            else:
                municipio = "São José do Rio Preto"
                uf = "SP"

            antes_cep = complemento_bairro_cep_municipio
            if cep_match:
                antes_cep = antes_cep[:cep_match.start()].strip()

            # Remove o trecho final do município/UF.
            antes_municipio = re.split(
                r"\b(?:SAO JOSE DO RIO PRETO|SÃO JOSÉ DO RIO PRETO)\s*-\s*IBGE\s*-\s*\d+",
                antes_cep,
                flags=re.IGNORECASE,
            )[0].strip()

            complemento_bairro = antes_municipio.strip()
            # Como complemento e bairro aparecem juntos, mantemos ambos na consulta.
            endereco = ", ".join(
                x for x in [logradouro_numero, complemento_bairro, municipio, uf, cep] if x
            )
            dados["ENDERECO_CNES"] = endereco

        # Algumas versões da ficha carregam latitude/longitude no JavaScript.
        # Se existirem, aproveitamos diretamente; caso contrário o endereço será geocodificado.
        padroes_lat = [
            r"(?:latitude|lat)[^\d-]{0,20}(-?\d{1,3}\.\d+)",
            r"(?:Latitude|LATITUDE)\s*[:=]\s*(-?\d{1,3}\.\d+)",
        ]
        padroes_lon = [
            r"(?:longitude|lng|lon)[^\d-]{0,20}(-?\d{1,3}\.\d+)",
            r"(?:Longitude|LONGITUDE)\s*[:=]\s*(-?\d{1,3}\.\d+)",
        ]
        for padrao in padroes_lat:
            achou = re.search(padrao, html, flags=re.IGNORECASE)
            if achou:
                dados["LATITUDE_CNES"] = float(achou.group(1))
                break
        for padrao in padroes_lon:
            achou = re.search(padrao, html, flags=re.IGNORECASE)
            if achou:
                dados["LONGITUDE_CNES"] = float(achou.group(1))
                break

        # Se a ficha não tiver endereço extraído, ainda retornamos a URL para diagnóstico.
        return dados

    except Exception as e:
        return {
            "CNES": cnes,
            "URL_CNES": url,
            "ENDERECO_CNES": None,
            "LATITUDE_CNES": np.nan,
            "LONGITUDE_CNES": np.nan,
            "STATUS_CNES_GEO": f"Erro consulta CNES: {type(e).__name__}",
        }


@st.cache_data(show_spinner=False)
def consultar_cadastro_cnes_em_lote(cnes_lista):
    """Busca os endereços oficiais do CNES uma única vez por estabelecimento."""
    resultados = []
    for cnes in list(cnes_lista):
        dados = consultar_cnes_oficial(cnes)
        if dados:
            resultados.append(dados)
    if not resultados:
        return pd.DataFrame(
            columns=[
                "CNES", "URL_CNES", "ENDERECO_CNES",
                "LATITUDE_CNES", "LONGITUDE_CNES", "STATUS_CNES_GEO"
            ]
        )
    return pd.DataFrame(resultados)


@st.cache_data(show_spinner=False, ttl=86400)
def baixar_unidades_saude_osm():
    """Baixa do OpenStreetMap, via Overpass, os estabelecimentos de saúde do município."""
    query = r"""
    [out:json][timeout:90];
    (
      nwr(around:20000,-20.8113,-49.3758)[amenity=hospital];
      nwr(around:20000,-20.8113,-49.3758)[amenity=clinic];
      nwr(around:20000,-20.8113,-49.3758)[amenity=doctors];
      nwr(around:20000,-20.8113,-49.3758)[healthcare];
    );
    out center tags;
    """
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    for endpoint in endpoints:
        try:
            resposta = requests.post(
                endpoint,
                data=query.encode("utf-8"),
                headers={"User-Agent": "Dashboard Dengue 2024 - OpenStreetMap"},
                timeout=120,
            )
            resposta.raise_for_status()
            elementos = resposta.json().get("elements", [])
            registros = []
            for el in elementos:
                tags = el.get("tags", {}) or {}
                nome = tags.get("name") or tags.get("official_name") or tags.get("alt_name")
                if not nome:
                    continue
                if "lat" in el and "lon" in el:
                    lat, lon = el["lat"], el["lon"]
                else:
                    center = el.get("center", {})
                    lat, lon = center.get("lat"), center.get("lon")
                if lat is None or lon is None:
                    continue
                tipo = tags.get("amenity") or tags.get("healthcare") or "saude"
                endereco_partes = []
                for chave in ["addr:street", "addr:housenumber", "addr:suburb", "addr:postcode"]:
                    if tags.get(chave):
                        endereco_partes.append(str(tags[chave]))
                registros.append({
                    "OSM_ID": f"{el.get('type','')}/{el.get('id','')}",
                    "OSM_NOME": str(nome).strip(),
                    "OSM_TIPO": str(tipo),
                    "LATITUDE_OSM": float(lat),
                    "LONGITUDE_OSM": float(lon),
                    "ENDERECO_OSM": ", ".join(endereco_partes),
                    "OSM_URL": f"https://www.openstreetmap.org/{el.get('type','')}/{el.get('id','')}",
                })
            df = pd.DataFrame(registros)
            if df.empty:
                return pd.DataFrame(columns=[
                    "OSM_ID", "OSM_NOME", "OSM_TIPO", "LATITUDE_OSM",
                    "LONGITUDE_OSM", "ENDERECO_OSM", "OSM_URL"
                ])
            return df.drop_duplicates(subset=["OSM_ID"])
        except Exception:
            continue
    return pd.DataFrame(columns=[
        "OSM_ID", "OSM_NOME", "OSM_TIPO", "LATITUDE_OSM",
        "LONGITUDE_OSM", "ENDERECO_OSM", "OSM_URL"
    ])


def normalizar_nome_osm(valor):
    import unicodedata
    texto = _texto_limpo(valor).upper()
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    texto = re.sub(r"[^A-Z0-9 ]+", " ", texto)
    texto = re.sub(r"\b(CNES|LTDA|ME|EIRELI|SA|MUNICIPAL|MUNICIPIO|PREFEITURA)\b", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def similaridade_nomes(a, b):
    from difflib import SequenceMatcher
    na, nb = normalizar_nome_osm(a), normalizar_nome_osm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    cobertura = len(ta & tb) / max(1, min(len(ta), len(tb)))
    return max(seq, 0.65 * seq + 0.35 * cobertura)


def encontrar_correspondencia_osm(nome, osm_df):
    if osm_df is None or osm_df.empty or not nome:
        return None
    candidatos = [(similaridade_nomes(nome, item.get("OSM_NOME", "")), item) for _, item in osm_df.iterrows()]
    candidatos.sort(key=lambda x: x[0], reverse=True)
    if not candidatos:
        return None
    melhor_score, melhor = candidatos[0]
    segundo_score = candidatos[1][0] if len(candidatos) > 1 else 0.0
    if melhor_score >= 0.86 or (melhor_score >= 0.74 and melhor_score - segundo_score >= 0.08):
        resultado = melhor.to_dict()
        resultado["SCORE_OSM"] = round(float(melhor_score), 3)
        return resultado
    return None


def geocodificar_endereco_osm(endereco):
    """Fallback exclusivamente OSM/Nominatim para nome/endereço textual."""
    if not endereco or str(endereco).strip() in {"", "nan", "None"}:
        return None
    try:
        local = geocode_nominatim(
            endereco,
            addressdetails=True,
            exactly_one=True,
            country_codes="br",
            limit=1,
        )
        if local:
            return {
                "lat": local.latitude,
                "lon": local.longitude,
                "endereco": getattr(local, "address", endereco),
                "status": "Geocodificado — OpenStreetMap/Nominatim",
            }
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def geocodificar_unidades(df_unidades, arquivo_base):
    """Usa o CNES da planilha para identificar a unidade e o OpenStreetMap para localizar."""
    base = df_unidades.copy()
    if "CNES" in base.columns:
        base["CNES"] = _normalizar_cnes(base["CNES"])

    try:
        cadastro = carregar_cadastro_cnes(arquivo_base)
        if not cadastro.empty and "CNES" in base.columns and "CNES" in cadastro.columns:
            cadastro_mapa = cadastro[[c for c in ["CNES", "ESTABELECIMENTO", "CNPJ", "GESTÃO"] if c in cadastro.columns]].copy()
            cadastro_mapa = cadastro_mapa.rename(columns={"ESTABELECIMENTO": "ESTABELECIMENTO_CNES", "CNPJ": "CNPJ_CNES", "GESTÃO": "GESTAO_CNES"})
            base = base.merge(cadastro_mapa, on="CNES", how="left")
            if "ESTABELECIMENTO_CNES" in base.columns:
                base["ESTABELECIMENTO"] = base["ESTABELECIMENTO_CNES"].fillna(base.get("ESTABELECIMENTO"))
    except Exception:
        pass

    base = base.drop_duplicates(subset=["CNES"] if "CNES" in base.columns else ["ESTABELECIMENTO"]).copy()
    for coluna, padrao in [("LATITUDE", np.nan), ("LONGITUDE", np.nan), ("STATUS_GEO", "Pendente"), ("ENDERECO_GEO", None), ("OSM_NOME", None), ("OSM_TIPO", None), ("OSM_ID", None), ("OSM_URL", None), ("SCORE_OSM", np.nan)]:
        if coluna not in base.columns:
            base[coluna] = padrao

    antigas = pd.DataFrame()
    if Path(ARQUIVO_COORDENADAS).exists():
        try:
            antigas = pd.read_csv(ARQUIVO_COORDENADAS, dtype={"CNES": "string"})
            if "CNES" in antigas.columns:
                antigas["CNES"] = _normalizar_cnes(antigas["CNES"])
        except Exception:
            antigas = pd.DataFrame()

    if not antigas.empty and "CNES" in antigas.columns and "CNES" in base.columns:
        cols = [c for c in ["CNES", "LATITUDE", "LONGITUDE", "ENDERECO_GEO", "OSM_NOME", "OSM_TIPO", "OSM_ID", "OSM_URL", "SCORE_OSM", "STATUS_GEO"] if c in antigas.columns]
        antigas = antigas[cols].drop_duplicates("CNES")
        base = base.merge(antigas, on="CNES", how="left", suffixes=("", "_ANTIGO"))
        for col in ["LATITUDE", "LONGITUDE", "ENDERECO_GEO", "OSM_NOME", "OSM_TIPO", "OSM_ID", "OSM_URL", "SCORE_OSM", "STATUS_GEO"]:
            antigo = f"{col}_ANTIGO"
            if antigo in base.columns:
                base[col] = base[col].combine_first(base[antigo])
                base.drop(columns=[antigo], inplace=True)

    osm_df = baixar_unidades_saude_osm()

    for idx, row in base.iterrows():
        lat = pd.to_numeric(row.get("LATITUDE"), errors="coerce")
        lon = pd.to_numeric(row.get("LONGITUDE"), errors="coerce")
        if pd.notna(lat) and pd.notna(lon):
            continue

        nome = _texto_limpo(row.get("ESTABELECIMENTO"))
        match = encontrar_correspondencia_osm(nome, osm_df)
        if match:
            base.at[idx, "LATITUDE"] = match["LATITUDE_OSM"]
            base.at[idx, "LONGITUDE"] = match["LONGITUDE_OSM"]
            base.at[idx, "OSM_NOME"] = match["OSM_NOME"]
            base.at[idx, "OSM_TIPO"] = match["OSM_TIPO"]
            base.at[idx, "OSM_ID"] = match["OSM_ID"]
            base.at[idx, "OSM_URL"] = match["OSM_URL"]
            base.at[idx, "SCORE_OSM"] = match["SCORE_OSM"]
            base.at[idx, "ENDERECO_GEO"] = match.get("ENDERECO_OSM", "")
            base.at[idx, "STATUS_GEO"] = "Localizado no OpenStreetMap por correspondência de nome"
            continue

        consultas = [
            f"{nome}, São José do Rio Preto, SP, Brasil",
            f"{nome}, São José do Rio Preto, São Paulo, Brasil",
        ] if nome else []
        for consulta in consultas:
            resultado = geocodificar_endereco_osm(consulta)
            if resultado:
                base.at[idx, "LATITUDE"] = resultado["lat"]
                base.at[idx, "LONGITUDE"] = resultado["lon"]
                base.at[idx, "ENDERECO_GEO"] = resultado["endereco"]
                base.at[idx, "OSM_NOME"] = nome
                base.at[idx, "OSM_TIPO"] = "Nominatim"
                base.at[idx, "STATUS_GEO"] = resultado["status"]
                break
        else:
            base.at[idx, "STATUS_GEO"] = "Não localizado no OpenStreetMap"

    colunas_salvar = [c for c in ["CNES", "ESTABELECIMENTO", "ENDERECO_GEO", "OSM_NOME", "OSM_TIPO", "OSM_ID", "OSM_URL", "SCORE_OSM", "LATITUDE", "LONGITUDE", "STATUS_GEO"] if c in base.columns]
    base[colunas_salvar].to_csv(ARQUIVO_COORDENADAS, index=False, encoding="utf-8-sig")
    return base

def criar_mapa_unidades(df_filtrado, arquivo_base):
    """Cria mapa interativo das unidades encontradas."""
    if df_filtrado.empty:
        return None, "Não existem registros para os filtros selecionados."

    colunas = ["CNES", "ESTABELECIMENTO"]
    colunas = [
        c for c in colunas
        if c in df_filtrado.columns
    ]

    unidades = (
        df_filtrado
        .groupby(colunas, dropna=False)
        .size()
        .reset_index(name="CASOS")
    )

    geo = geocodificar_unidades(unidades, arquivo_base)

    geo["LATITUDE"] = pd.to_numeric(
        geo["LATITUDE"],
        errors="coerce"
    )

    geo["LONGITUDE"] = pd.to_numeric(
        geo["LONGITUDE"],
        errors="coerce"
    )

    geo = geo[
        geo["LATITUDE"].notna()
        & geo["LONGITUDE"].notna()
    ].copy()

    if geo.empty:
        return None, (
            "Nenhuma unidade possui coordenadas. "
            "A base precisa conter endereço/logradouro "
            "ou um arquivo coordenadas_cnes.csv."
        )

    mapa = folium.Map(
        location=[-20.8113, -49.3758],
        zoom_start=12,
        control_scale=True
    )

    for _, row in geo.iterrows():

        casos = int(row["CASOS"])

        # Marcadores maiores para unidades com mais notificações
        raio = max(
            6,
            min(
                18,
                5 + np.log1p(casos) * 2
            )
        )

        estabelecimento = str(
            row.get(
                "ESTABELECIMENTO",
                "Unidade"
            )
        )

        cnes = str(
            row.get(
                "CNES",
                "N/D"
            )
        )

        status_geo = str(
            row.get(
                "STATUS_GEO",
                "N/D"
            )
        )

        popup = f"""
        <div style="font-size:14px">
            <b>{estabelecimento}</b><br>
            CNES: {cnes}<br>
            Notificações: {casos:,}<br>
            Geocodificação: {status_geo}
        </div>
        """

        folium.CircleMarker(
            location=[
                float(row["LATITUDE"]),
                float(row["LONGITUDE"])
            ],
            radius=raio,
            popup=folium.Popup(
                popup,
                max_width=350
            ),
            tooltip=(
                f"{estabelecimento} — "
                f"{casos} notificações"
            ),
            fill=True
        ).add_to(mapa)

    return mapa, geo


def aplicar_filtros(df):
    st.sidebar.header("🔎 Filtros")

    meses = ["Todos"] + [
        "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
        "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
    ]
    mes = st.sidebar.selectbox("Mês", meses)

    trimestres = ["Todos"] + sorted(df["TRIMESTRE"].dropna().unique().tolist())
    trimestre = st.sidebar.selectbox("Trimestre", trimestres)

    sexos = ["Todos"] + sorted(df["SEXO_DESC"].dropna().unique().tolist())
    sexo = st.sidebar.selectbox("Sexo", sexos)

    faixas_ordem = [
        "0–9", "10–19", "20–29", "30–39",
        "40–49", "50–59", "60+", "Sem informação"
    ]
    faixas_disponiveis = [x for x in faixas_ordem if x in set(df["FAIXA_ETARIA"])]
    faixa = st.sidebar.selectbox("Faixa etária", ["Todas"] + faixas_disponiveis)

    sorotipos = ["Todos"] + sorted(df["SOROTIPO_DESC"].dropna().unique().tolist())
    sorotipo = st.sidebar.selectbox("Sorotipo (código)", sorotipos)

    sintomas_filtro = ["Todos"] + list(SINTOMAS.values())
    sintoma = st.sidebar.selectbox("Sintoma presente", sintomas_filtro)

    unidades = sorted(df["ESTABELECIMENTO"].dropna().astype(str).unique().tolist())
    unidade = st.sidebar.selectbox(
        "Unidade de saúde",
        ["Todas"] + unidades,
    )

    gestoes = ["Todos"] + sorted(df["GESTAO"].dropna().astype(str).unique().tolist())
    gestao = st.sidebar.selectbox("Gestão", gestoes)

    status = ["Todos"] + sorted(df["STATUS_CNES"].dropna().astype(str).unique().tolist())
    status_cnes = st.sidebar.selectbox("Integração CNES", status)

    qtd = st.sidebar.selectbox(
        "Quantidade de sintomas",
        ["Todas", "0", "1", "2", "3–4", "5+"]
    )

    out = df.copy()

    if mes != "Todos":
        out = out[out["MES"] == mes]
    if trimestre != "Todos":
        out = out[out["TRIMESTRE"] == trimestre]
    if sexo != "Todos":
        out = out[out["SEXO_DESC"] == sexo]
    if faixa != "Todas":
        out = out[out["FAIXA_ETARIA"] == faixa]
    if sorotipo != "Todos":
        out = out[out["SOROTIPO_DESC"] == sorotipo]
    if unidade != "Todas":
        out = out[out["ESTABELECIMENTO"].astype(str) == unidade]
    if gestao != "Todos":
        out = out[out["GESTAO"].astype(str) == gestao]
    if status_cnes != "Todos":
        out = out[out["STATUS_CNES"].astype(str) == status_cnes]

    if qtd != "Todas":
        if qtd in ["0", "1", "2"]:
            out = out[out["QTD_SINTOMAS"] == int(qtd)]
        elif qtd == "3–4":
            out = out[out["QTD_SINTOMAS"].between(3, 4)]
        elif qtd == "5+":
            out = out[out["QTD_SINTOMAS"] >= 5]

    if sintoma != "Todos":
        coluna = next((k for k, v in SINTOMAS.items() if v == sintoma), None)
        if coluna:
            out = out[out[coluna] == 1]

    return out


def grafico_vazio(titulo, mensagem):
    fig = go.Figure()
    fig.update_layout(
        title=titulo,
        annotations=[{
            "text": mensagem,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5,
            "showarrow": False,
        }],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


# ============================================================
# CARREGAMENTO
# ============================================================
st.title("🦟 Dashboard Epidemiológico da Dengue — 2024")
st.caption("São José do Rio Preto — análise de notificações integradas ao CNES")

uploaded = st.sidebar.file_uploader(
    "Opcional: carregue outra base Excel",
    type=["xlsx"],
)

if uploaded is not None:
    df = carregar_base(uploaded)
else:
    try:
        df = carregar_base(ARQUIVO_PADRAO)
    except FileNotFoundError:
        st.error(
            f"Arquivo '{ARQUIVO_PADRAO.name}' não encontrado. "
            "Coloque o Excel na mesma pasta do app.py ou carregue-o pela barra lateral."
        )
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar a base: {e}")
        st.stop()

df_filtrado = aplicar_filtros(df)

if st.sidebar.button("🔄 Atualizar geolocalização", help="Apaga o cache e consulta novamente o OpenStreetMap/Overpass e o Nominatim."):
    try:
        if Path(ARQUIVO_COORDENADAS).exists():
            Path(ARQUIVO_COORDENADAS).unlink()
    except Exception:
        pass
    geocodificar_unidades.clear()
    baixar_unidades_saude_osm.clear()
    st.rerun()

# ============================================================
# KPIs
# ============================================================
total = len(df_filtrado)
idade_media = df_filtrado["IDADE"].mean()
unidades = df_filtrado["ESTABELECIMENTO"].nunique()
cnes_encontrado = (df_filtrado["STATUS_CNES"] == "Encontrado no CNES").sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Casos/notificações", f"{total:,}".replace(",", "."))
k2.metric(
    "Idade média",
    f"{idade_media:.1f} anos" if pd.notna(idade_media) else "N/D"
)
k3.metric("Unidades de saúde", f"{unidades:,}".replace(",", "."))
k4.metric("Registros com CNES", f"{cnes_encontrado:,}".replace(",", "."))
k5.metric(
    "Cobertura CNES",
    f"{(cnes_encontrado / total * 100):.2f}%" if total else "0%"
)

st.divider()

# ============================================================
# POPULAÇÃO / INCIDÊNCIA
# ============================================================
with st.expander("📌 Incidência — configuração da população"):
    st.write(
        "A base fornecida contém notificações, mas não possui o denominador populacional. "
        "Por isso, a incidência não é inventada: informe a população de referência para calcular "
        "casos por 100.000 habitantes."
    )
    populacao = st.number_input(
        "População de referência do município",
        min_value=0,
        value=0,
        step=1000,
    )
    if populacao > 0:
        incidencia = total / populacao * 100000
        st.metric("Incidência estimada (casos/100 mil hab.)", f"{incidencia:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

# ============================================================
# ABAS
# ============================================================
(
    aba_geral,
    aba_geo,
    aba_demografico,
    aba_sintomas,
    aba_relacao,
    aba_saude,
    aba_dados,
) = st.tabs([
    "📊 Visão Geral",
    "🗺️ Distribuição Geográfica",
    "👥 Perfil Demográfico",
    "🤒 Sintomas",
    "🔗 Relação entre Sintomas",
    "🏥 Atendimento e CNES",
    "📋 Dados",
])

# ============================================================
# 1. VISÃO GERAL
# ============================================================
with aba_geral:
    st.subheader("3.5.1 — Visão Geral dos Casos de Dengue")

    mensal = (
        df_filtrado.groupby(["MES_NUM", "MES"], dropna=False)
        .size()
        .reset_index(name="Casos")
        .sort_values("MES_NUM")
    )

    fig = px.line(
        mensal,
        x="MES",
        y="Casos",
        markers=True,
        title="Evolução temporal das notificações em 2024",
        category_orders={"MES": [
            "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
            "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
        ]},
    )
    st.plotly_chart(fig, use_container_width=True, key="grafico_evolucao_mensal")

    c1, c2 = st.columns(2)

    with c1:
        trim = df_filtrado["TRIMESTRE"].value_counts().rename_axis("Trimestre").reset_index(name="Casos")
        trim["ordem"] = trim["Trimestre"].str.extract(r"(\d)").astype(float)
        trim = trim.sort_values("ordem")
        fig_trim = px.bar(trim, x="Trimestre", y="Casos", title="Casos por trimestre")
        st.plotly_chart(fig_trim, use_container_width=True, key="grafico_casos_trimestre")

    with c2:
        sexo_df = df_filtrado["SEXO_DESC"].value_counts().rename_axis("Sexo").reset_index(name="Casos")
        fig_sexo = px.pie(sexo_df, names="Sexo", values="Casos", hole=0.35, title="Distribuição por sexo")
        st.plotly_chart(fig_sexo, use_container_width=True, key="grafico_distribuicao_sexo_geral")

    st.info(
        "Interpretação: os gráficos permitem identificar os períodos de maior concentração "
        "das notificações. A concentração temporal descreve o comportamento observado, "
        "mas não estabelece sozinha uma relação causal."
    )

# ============================================================
# 2. GEOGRÁFICA
# ============================================================
with aba_geo:
    st.subheader("3.5.2 — Distribuição Geográfica dos Casos")

    st.write(
        "Mapa interativo das unidades de saúde associadas às notificações. "
        "O aplicativo utiliza o CNES que já existe na planilha para identificar "
        "a unidade e consulta o OpenStreetMap. Primeiro, o sistema busca os "
        "estabelecimentos de saúde mapeados no OSM via Overpass e faz a correspondência "
        "pelo nome da unidade. Se não encontrar, usa o Nominatim, também do OSM, como fallback. "
        "Nenhuma coordenada é inventada."
    )

    arquivo_mapa = uploaded if uploaded is not None else ARQUIVO_PADRAO
    mapa, resultado_geo = criar_mapa_unidades(
        df_filtrado, arquivo_mapa
    )

    if mapa is not None:

        st_folium(
            mapa,
            use_container_width=True,
            height=620,
            returned_objects=[]
        )

        if isinstance(resultado_geo, pd.DataFrame):

            m1, m2 = st.columns(2)

            m1.metric(
                "Unidades geolocalizadas",
                f"{len(resultado_geo):,}".replace(",", ".")
            )

            m2.metric(
                "Notificações representadas",
                f"{resultado_geo['CASOS'].sum():,}".replace(",", ".")
            )

    else:
        st.warning(resultado_geo)

    st.markdown("### Distribuição por unidade")

    ranking_geo = (
        df_filtrado["ESTABELECIMENTO"]
        .value_counts()
        .head(30)
        .rename_axis("Estabelecimento")
        .reset_index(name="Casos")
    )

    fig_geo = px.bar(
        ranking_geo.sort_values("Casos"),
        x="Casos",
        y="Estabelecimento",
        orientation="h",
        title="Top 30 unidades por número de notificações"
    )

    st.plotly_chart(
        fig_geo,
        use_container_width=True,
        key="grafico_top_unidades_geografica"
    )

    with st.expander("ℹ️ Informações sobre a geocodificação"):

        st.write(
            "A geocodificação é realizada somente para unidades "
            "únicas pelo CNES, e não para cada uma das notificações. "
            "As coordenadas encontradas são armazenadas no arquivo "
            "coordenadas_cnes.csv para evitar consultas repetidas."
        )

        st.code(
            """Arquivo gerado:
coordenadas_cnes.csv

Colunas:
CNES
ESTABELECIMENTO
ENDERECO_GEO
LATITUDE
LONGITUDE
STATUS_GEO""",
            language="text"
        )

# ============================================================
# 3. DEMOGRÁFICO
# ============================================================
with aba_demografico:
    st.subheader("3.5.3 — Perfil Demográfico")

    c1, c2 = st.columns(2)

    with c1:
        faixa_ordem = ["0–9", "10–19", "20–29", "30–39", "40–49", "50–59", "60+", "Sem informação"]
        faixa_df = df_filtrado["FAIXA_ETARIA"].value_counts().rename_axis("Faixa etária").reset_index(name="Casos")
        faixa_df["ordem"] = faixa_df["Faixa etária"].map({x: i for i, x in enumerate(faixa_ordem)})
        faixa_df = faixa_df.sort_values("ordem")
        fig_faixa = px.bar(faixa_df, x="Faixa etária", y="Casos", title="Casos por faixa etária")
        st.plotly_chart(fig_faixa, use_container_width=True, key="grafico_casos_faixa_etaria")

    with c2:
        sexo_df = df_filtrado["SEXO_DESC"].value_counts().rename_axis("Sexo").reset_index(name="Casos")
        fig_sexo2 = px.pie(sexo_df, names="Sexo", values="Casos", title="Distribuição por sexo", hole=0.35)
        st.plotly_chart(fig_sexo2, use_container_width=True, key="grafico_distribuicao_sexo_demografico")

    idade_valida = df_filtrado["IDADE"].dropna()
    if not idade_valida.empty:
        fig_hist = px.histogram(
            idade_valida,
            x="IDADE",
            nbins=30,
            title="Distribuição das idades",
            labels={"IDADE": "Idade"},
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="grafico_histograma_idades")

# ============================================================
# 4. SINTOMAS
# ============================================================
with aba_sintomas:
    st.subheader("3.5.4 — Análise de Sintomas")

    linhas = []
    for col, nome in SINTOMAS.items():
        linhas.append({
            "Sintoma": nome,
            "Casos": int((df_filtrado[col] == 1).sum()),
        })
    sint_df = pd.DataFrame(linhas).sort_values("Casos", ascending=False)

    c1, c2 = st.columns(2)

    with c1:
        fig_rank = px.bar(
            sint_df.sort_values("Casos"),
            x="Casos",
            y="Sintoma",
            orientation="h",
            title="Ranking de sintomas",
        )
        st.plotly_chart(fig_rank, use_container_width=True, key="grafico_ranking_sintomas")

    with c2:
        fig_pie = px.pie(
            sint_df,
            names="Sintoma",
            values="Casos",
            title="Participação dos registros sintomáticos",
            hole=0.25,
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="grafico_participacao_sintomas")

    st.markdown("### Sintomas por sexo")
    sexo_sint = []
    for col, nome in SINTOMAS.items():
        for sexo_nome in ["Feminino", "Masculino", "Ignorado"]:
            sub = df_filtrado[df_filtrado["SEXO_DESC"] == sexo_nome]
            sexo_sint.append({
                "Sintoma": nome,
                "Sexo": sexo_nome,
                "Casos": int((sub[col] == 1).sum()),
            })

    sexo_sint_df = pd.DataFrame(sexo_sint)
    fig_sexo_sint = px.bar(
        sexo_sint_df,
        x="Sintoma",
        y="Casos",
        color="Sexo",
        barmode="group",
        title="Comparação de sintomas por sexo",
    )
    st.plotly_chart(fig_sexo_sint, use_container_width=True, key="grafico_sintomas_por_sexo")

    st.markdown("### Sintomas por faixa etária")
    faixa_sint = []
    for col, nome in SINTOMAS.items():
        for faixa_nome in ["0–9", "10–19", "20–29", "30–39", "40–49", "50–59", "60+"]:
            sub = df_filtrado[df_filtrado["FAIXA_ETARIA"] == faixa_nome]
            faixa_sint.append({
                "Sintoma": nome,
                "Faixa etária": faixa_nome,
                "Casos": int((sub[col] == 1).sum()),
            })

    faixa_sint_df = pd.DataFrame(faixa_sint)
    fig_faixa_sint = px.bar(
        faixa_sint_df,
        x="Faixa etária",
        y="Casos",
        color="Sintoma",
        barmode="group",
        title="Sintomas por faixa etária",
    )
    st.plotly_chart(fig_faixa_sint, use_container_width=True, key="grafico_sintomas_por_faixa")

# ============================================================
# 5. RELAÇÃO ENTRE SINTOMAS
# ============================================================
with aba_relacao:
    st.subheader("3.5.5 — Relação entre Sintomas")

    st.write(
        "A matriz abaixo usa associação entre variáveis binárias (presença/ausência do sintoma). "
        "Ela é exploratória e não deve ser interpretada como causalidade."
    )

    matriz = df_filtrado[list(SINTOMAS.keys())].copy()
    matriz.columns = [SINTOMAS[c] for c in matriz.columns]
    matriz = matriz.replace({2: 0})
    matriz = matriz.apply(pd.to_numeric, errors="coerce")
    corr = matriz.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Matriz de associação entre sintomas",
    )
    st.plotly_chart(fig_corr, use_container_width=True, key="grafico_matriz_associacao_sintomas")

    st.markdown("### Sintomas segundo o sorotipo informado")
    sor_df = []
    for sor, sub in df_filtrado.groupby("SOROTIPO_DESC"):
        for col, nome in SINTOMAS.items():
            sor_df.append({
                "Sorotipo": sor,
                "Sintoma": nome,
                "Casos": int((sub[col] == 1).sum()),
            })

    sor_df = pd.DataFrame(sor_df)
    if not sor_df.empty:
        fig_sor = px.bar(
            sor_df,
            x="Sintoma",
            y="Casos",
            color="Sorotipo",
            barmode="group",
            title="Frequência dos sintomas por código de sorotipo",
        )
        st.plotly_chart(fig_sor, use_container_width=True, key="grafico_sintomas_por_sorotipo")

# ============================================================
# 6. ATENDIMENTO / CNES
# ============================================================
with aba_saude:
    st.subheader("3.5.6 — Encaminhamento e Atendimento em Saúde")

    unidade_df = (
        df_filtrado["ESTABELECIMENTO"]
        .value_counts()
        .head(30)
        .rename_axis("Estabelecimento")
        .reset_index(name="Atendimentos/Notificações")
    )

    fig_unid = px.bar(
        unidade_df.sort_values("Atendimentos/Notificações"),
        x="Atendimentos/Notificações",
        y="Estabelecimento",
        orientation="h",
        title="30 estabelecimentos com maior número de notificações",
    )
    st.plotly_chart(fig_unid, use_container_width=True, key="grafico_top_estabelecimentos")

    c1, c2 = st.columns(2)

    with c1:
        gestao_df = (
            df_filtrado["GESTAO"]
            .fillna("Sem informação")
            .value_counts()
            .rename_axis("Gestão")
            .reset_index(name="Casos")
        )
        fig_gestao = px.pie(gestao_df, names="Gestão", values="Casos", hole=0.35, title="Casos por gestão")
        st.plotly_chart(fig_gestao, use_container_width=True, key="grafico_casos_por_gestao")

    with c2:
        status_df = (
            df_filtrado["STATUS_CNES"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Registros")
        )
        fig_status = px.pie(status_df, names="Status", values="Registros", hole=0.35, title="Integração com CNES")
        st.plotly_chart(fig_status, use_container_width=True, key="grafico_integracao_cnes")

    st.info(
        "A base não contém uma variável explícita de hospitalização ou encaminhamento. "
        "Assim, esta tela analisa notificações por estabelecimento e a integração com CNES, "
        "sem afirmar que todos os registros representam internações."
    )

# ============================================================
# 7. DADOS
# ============================================================
with aba_dados:
    st.subheader("📋 Base filtrada")

    st.write(f"Registros exibidos: **{len(df_filtrado):,}**".replace(",", "."))

    colunas_mostrar = [
        "ID_REGISTRO", "DT_NOTIFIC", "ID_MUNICIP", "ID_UNIDADE",
        "ESTABELECIMENTO", "CNES", "STATUS_CNES", "SEXO_DESC",
        "IDADE", "FAIXA_ETARIA", "SOROTIPO_DESC",
        "QTD_SINTOMAS", "TRIMESTRE"
    ]
    colunas_mostrar = [c for c in colunas_mostrar if c in df_filtrado.columns]

    st.dataframe(
        df_filtrado[colunas_mostrar],
        use_container_width=True,
        height=500,
    )

    csv = df_filtrado.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Baixar base filtrada em CSV",
        data=csv,
        file_name="dengue_2024_base_filtrada.csv",
        mime="text/csv",
    )

# ============================================================
# RODAPÉ / LIMITAÇÕES
# ============================================================
st.divider()
st.caption(
    "Fonte: base epidemiológica de Dengue 2024 integrada ao CNES fornecida para o projeto. "
    "Filtros e indicadores são recalculados dinamicamente."
)

with st.expander("⚠️ Limitações metodológicas da base atual"):
    st.markdown(
        """
- A base possui **41.928 registros** e 23 colunas na planilha BASE_COMPLETA.
- A integração com o CNES está disponível por **ID_UNIDADE → CNES**.
- A base epidemiológica não possui latitude/longitude e a aba BASE_CNES não traz o endereço. O mapa utiliza o CNES já existente na planilha para consultar a ficha pública do CNES/DATASUS, obter o endereço oficial e então gerar as coordenadas; unidades que não puderem ser geocodificadas não são exibidas.
- A base atual não possui um campo explícito de hospitalização/encaminhamento; portanto o dashboard não inventa esse indicador.
- A incidência depende de um denominador populacional externo. O dashboard permite informar a população e calcula casos por 100.000 habitantes.
- O campo SOROTIPO possui códigos informados na própria base. O dashboard mantém os códigos para evitar atribuição incorreta de significados sem o dicionário original.
- A relação entre sintomas é apresentada como associação exploratória, não como causalidade.
"""
    )