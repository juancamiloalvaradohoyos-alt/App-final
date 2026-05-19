"""
🏥 Sistema de Predicción de Servicios de Salud – Población Migrante Bogotá
Analítica 3 | ODS 3 · ODS 10 · ODS 11
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
import re
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Salud Migrante Bogotá · Predicción IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .hero-header {
    background: linear-gradient(135deg, #1565c0 0%, #0288d1 60%, #00acc1 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.2rem;
    text-align: center;
  }
  .hero-title { font-size: 2rem; font-weight: 800; color: #ffffff !important; margin: 0; text-shadow: 0 2px 8px rgba(0,0,0,0.3); }
  .hero-subtitle { font-size: 0.95rem; color: rgba(255,255,255,0.88) !important; margin-top: 0.5rem; }
  .ods-row { text-align: center; margin-bottom: 1rem; }
  .ods-pill { display: inline-block; border-radius: 20px; padding: 5px 16px; font-size: 0.82rem; font-weight: 700; margin: 3px; color: #ffffff !important; }
  .ods-3  { background: #2e7d32; }
  .ods-10 { background: #6a1b9a; }
  .ods-11 { background: #e65100; }
  .kpi-card { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.13); border-radius: 14px; padding: 1.2rem 0.8rem; text-align: center; border-top: 4px solid #29b6f6; }
  .kpi-value { font-size: 1.9rem; font-weight: 800; color: #29b6f6 !important; line-height: 1; }
  .kpi-label { font-size: 0.72rem; color: rgba(255,255,255,0.60) !important; margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }
  .pred-box { border-radius: 16px; padding: 2rem 1.5rem; text-align: center; margin-top: 1.2rem; border: 2px solid; }
  .pred-icon { font-size: 3.5rem; line-height: 1; }
  .pred-service { font-size: 1.9rem; font-weight: 800; margin: 0.6rem 0 0.2rem; }
  .pred-conf { font-size: 0.95rem; opacity: 0.9; font-weight: 600; }
  .prob-row { margin: 7px 0; }
  .prob-label { font-size: 0.88rem; font-weight: 600; color: rgba(255,255,255,0.85) !important; margin-bottom: 3px; }
  .prob-bar-bg { background: rgba(255,255,255,0.10); border-radius: 6px; height: 13px; width: 100%; overflow: hidden; }
  .prob-bar-fill { height: 13px; border-radius: 6px; }
  .sec-header { font-size: 1.2rem; font-weight: 800; color: #29b6f6 !important; border-bottom: 2px solid #0288d1; padding-bottom: 5px; margin: 1.2rem 0 0.8rem; }
  .resumen-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
  .resumen-table th { background: rgba(41,182,246,0.20); color: #29b6f6 !important; padding: 7px 10px; text-align:left; font-weight:700; }
  .resumen-table td { padding: 7px 10px; color: rgba(255,255,255,0.82) !important; border-bottom: 1px solid rgba(255,255,255,0.07); }
  .rec-box { background: rgba(41,182,246,0.10); border: 1px solid rgba(41,182,246,0.35); border-radius: 12px; padding: 1rem 1.2rem; margin-top: 1rem; color: rgba(255,255,255,0.88) !important; font-size: 0.93rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ──
plt.rcParams.update({
    "figure.facecolor": "#1a1f2e",
    "axes.facecolor":   "#1a1f2e",
    "axes.edgecolor":   "#3a4560",
    "axes.labelcolor":  "#b0bec5",
    "axes.titlecolor":  "#e8f0fe",
    "xtick.color":      "#90a4ae",
    "ytick.color":      "#90a4ae",
    "text.color":       "#cfd8e3",
    "grid.color":       "#283040",
    "grid.linestyle":   "--",
    "grid.alpha":       0.55,
    "legend.facecolor": "#1a1f2e",
    "legend.edgecolor": "#3a4560",
    "font.size":        9.5,
})

# ── Constantes ──
DATA_PATH  = os.path.join(os.path.dirname(__file__), "Base_de_Datos_PA.xlsx")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pipe.pkl")
MODEL_PATH = os.path.join(tempfile.gettempdir(), "model_pipe.pkl")
LE_PATH    = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")
MODEL_FILE_ID = "1MU1KGeaLLP2mqjrhW_51RZka-RHNBaT_"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

MESES = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
          7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
ANIO_OPTS = [2022, 2023, 2024, 2025, 2026]

SEXO_OPTS       = ["Femenino","Masculino"]
CURSO_VIDA_OPTS = ["0-6 años","7-11 años","12-17 años","18-28 años","29-59 años","60+ años"]
LUGAR_OPTS = ["CS BUCARAMANGA","CS CAFE MADRID","CS CAMPO HERMOSO","CS COLORADOS",
    "CS COMUNEROS","CS CONCORDIA","CS CRISTAL","CS GAITAN","CS GIRARDOT",
    "CS IPC","CS KENNEDY","CS LA JOYA","CS LIBERTAD","CS MORRORICO",
    "CS MUTIS","CS PABLO VI","CS REGADEROS","CS ROSARIO","CS SAN RAFAEL",
    "CS SANTANDER","CS TOLEDO PLATA","CS VILLA ROSA","EQUIPO EXTRAMURAL",
    "EXTRAMURAL","HOSPITAL LOCAL NORTE","UIMIST","UNIDAD MOVIL"]
ESPECIALIDAD_OPTS = ["ANESTESIOLOGIA","AUX. DE ENFERMERIA","CIRUGIA GENERAL","DERMATOLOGIA",
    "ENFERMERIA PROFESIONAL","FISIOTERAPIA","FONOAUDIOLOGIA","GINECOLOGIA Y OBSTETRICIA",
    "HIGIENE ORAL","IMAGENES DIAGNOSTICAS","LABORATORIO CLINICO","MEDICINA GENERAL",
    "MEDICINA INTERNA","NUTRICION","ODONTOLOGIA GENERAL","PEDIATRIA","PSICOLOGIA",
    "TERAPIA RESPIRATORIA","TRABAJO SOCIAL","TRANSPORTE ASISTENCIAL MEDICALIZADO"]

SMETA = {
    "CONSULTA":    {"icon":"🩺","color":"#29b6f6","bg":"rgba(41,182,246,0.13)", "label":"Consulta Médica"},
    "VACUNACION":  {"icon":"💉","color":"#66bb6a","bg":"rgba(102,187,106,0.13)","label":"Vacunación"},
    "LABORATORIO": {"icon":"🔬","color":"#ce93d8","bg":"rgba(206,147,216,0.13)","label":"Laboratorio"},
    "IMAGEN":      {"icon":"🩻","color":"#ffb74d","bg":"rgba(255,183,77,0.13)", "label":"Imagen Diagnóstica"},
    "TERAPIA":     {"icon":"🧘","color":"#4dd0e1","bg":"rgba(77,208,225,0.13)", "label":"Terapia"},
    "OTROS":       {"icon":"🏥","color":"#90a4ae","bg":"rgba(144,164,174,0.13)","label":"Otros Servicios"},
}

ANA = {
    "total":47315,"lugares":27,
    "servicios_dist":{"CONSULTA":18439,"OTROS":15314,"VACUNACION":10603,"LABORATORIO":2081,"IMAGEN":859,"TERAPIA":19},
    "por_anio":{2022:25457,2023:13059,2024:8799},
    "por_sexo":{"Femenino":32900,"Masculino":14415},
    "curso_vida":{"18-28 años":18945,"29-59 años":12987,"7-11 años":8515,"0-6 años":6868},
    "top_esp":{"LAB. CLINICO":12076,"MEDICINA GENERAL":11505,"AUX. ENFERMERIA":10708,
               "PSICOLOGIA":5722,"ENFERMERIA PROF.":1972,"ODONTOLOGIA":1804,
               "GINECOLOGIA":1271,"NUTRICION":1087,"HIGIENE ORAL":611,"IMAG. DIAG.":297},
    "top_lug":{"UIMIST":10299,"CS MORRORICO":9103,"CS COMUNEROS":8833,"EXTRAMURAL":8260,
               "HOSP. LOCAL NORTE":3024,"CS SANTANDER":993,"CS GIRARDOT":961,
               "CS CAFE MADRID":907,"CS BUCARAMANGA":611,"CS GAITAN":568},
    "valor_prom":{"CONSULTA":36542,"IMAGEN":123000,"LABORATORIO":30904,
                  "OTROS":70455,"TERAPIA":31528,"VACUNACION":23},
}

# ── Helpers ──
def normalize_curso_vida(s):
    s = str(s).strip()
    try: s = s.encode("latin1").decode("utf-8")
    except: pass
    if "0 a" in s or "0a" in s: return "0-6 años"
    if "7" in s or "6 a 11" in s: return "7-11 años"
    if "12" in s: return "12-17 años"
    if "18" in s: return "18-28 años"
    if "29" in s: return "29-59 años"
    if "60" in s or ">60" in s: return "60+ años"
    return "Otro"

def normalize_lugar(s):
    s = str(s).strip()
    s = re.sub(r"^C\.S\s+","CS ",s)
    s = re.sub(r"^SEDE\s+","",s)
    return s.strip()

def agrupar_servicio(servicio):
    s = str(servicio).lower()
    if "consulta" in s: return "CONSULTA"
    elif "vacuna" in s or "vacun" in s: return "VACUNACION"
    elif "terapia" in s: return "TERAPIA"
    elif "hemo" in s or "laboratorio" in s: return "LABORATORIO"
    elif "ultra" in s or "radio" in s or "eco" in s: return "IMAGEN"
    else: return "OTROS"

def dark_fig(w=7, h=4.2):
    fig, ax = plt.subplots(figsize=(w,h), facecolor="#1a1f2e")
    ax.set_facecolor("#1a1f2e")
    return fig, ax

def download_model_if_needed():
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        downloaded_path = gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
        if downloaded_path is None or not os.path.exists(MODEL_PATH):
            raise RuntimeError("No se pudo descargar model_pipe.pkl desde Google Drive.")
    return MODEL_PATH

def load_label_encoder():
    if os.path.exists(LE_PATH):
        return joblib.load(LE_PATH)
    le = LabelEncoder()
    le.fit(["CONSULTA", "IMAGEN", "LABORATORIO", "OTROS", "TERAPIA", "VACUNACION"])
    return le

@st.cache_resource(show_spinner="⚙️ Cargando modelo de IA…")
def load_model():
    model_path = download_model_if_needed()
    if os.path.exists(model_path):
        return joblib.load(model_path), load_label_encoder()
    df = pd.read_excel(DATA_PATH)
    df = df[(df["AÑO"]>=2022)&(df["AÑO"]<=2024)].copy()
    df["Curso de vida"] = df["Curso de vida"].astype(str).apply(normalize_curso_vida)
    df["LUGAR"]         = df["LUGAR"].astype(str).apply(normalize_lugar)
    df["MES_NUM"]       = df["MES"].astype(str).str.extract(r"(\d{1,2})").astype(float).fillna(1).astype(int)
    df["SERVICIO_TIPO"] = df["SERVICIO"].apply(agrupar_servicio)
    df = df.dropna(subset=["Sexo","ESPECIALIDAD","LUGAR"])
    cat_cols=["Sexo","Curso de vida","LUGAR","ESPECIALIDAD"]
    num_cols=["AÑO","MES_NUM"]
    X=df[cat_cols+num_cols]; le=LabelEncoder(); y=le.fit_transform(df["SERVICIO_TIPO"])
    pre=ColumnTransformer([("n",StandardScaler(),num_cols),("c",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat_cols)])
    pipe=Pipeline([("pre",pre),("clf",RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1))])
    pipe.fit(X,y)
    joblib.dump(pipe,MODEL_PATH); joblib.dump(le,LE_PATH)
    return pipe,le

# ── Sidebar ──
with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=72)
    st.markdown("### 🏥 Panel de Control")
    st.markdown("---")
    seccion = st.radio("Navegación",[
        "🏠 Inicio","📊 Analítica del Sistema",
        "🔮 Predicción de Servicio","📋 Acerca del Proyecto"])
    st.markdown("---")
    st.caption("🇨🇴 Bogotá · Analítica 3")
    st.caption("ODS 3 · ODS 10 · ODS 11")

# ── Header global ──
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🏥 Predicción de Servicios de Salud</div>
  <div class="hero-subtitle">Población Migrante · Bogotá, Colombia · 2022–2024 · IA con Random Forest</div>
</div>
<div class="ods-row">
  <span class="ods-pill ods-3">🟢 ODS 3 – Salud y Bienestar</span>
  <span class="ods-pill ods-10">🟣 ODS 10 – Reducción de Desigualdades</span>
  <span class="ods-pill ods-11">🟠 ODS 11 – Ciudades Sostenibles</span>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════
if seccion == "🏠 Inicio":
    st.markdown('<div class="sec-header">📌 Indicadores Clave</div>', unsafe_allow_html=True)
    kpis=[("47,315","Atenciones 2022-2024"),("27","Centros de atención"),
          ("6","Tipos de servicio"),("79%","Precisión del modelo"),("3","Años analizados")]
    for col,(val,lbl) in zip(st.columns(5),kpis):
        col.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    cl,cr=st.columns(2)

    with cl:
        st.markdown('<div class="sec-header">📊 Distribución de Servicios</div>',unsafe_allow_html=True)
        sd=ANA["servicios_dist"]
        lbls=[SMETA[k]["label"] for k in sd]; vals=list(sd.values()); clrs=[SMETA[k]["color"] for k in sd]
        fig,ax=dark_fig(7,4.2)
        bars=ax.barh(lbls,vals,color=clrs,edgecolor="none",height=0.62)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#3a4560")
        ax.set_xlabel("Atenciones",color="#90a4ae",fontsize=9)
        ax.tick_params(axis="y",labelsize=9,labelcolor="#dce8f0")
        ax.tick_params(axis="x",labelcolor="#90a4ae")
        ax.set_xlim(0,max(vals)*1.23)
        ax.xaxis.grid(True,alpha=0.35)
        ax.set_axisbelow(True)
        for bar,val in zip(bars,vals):
            ax.text(bar.get_width()+max(vals)*0.012, bar.get_y()+bar.get_height()/2,
                    f"{val:,}",va="center",fontsize=8.5,color="#dce8f0",fontweight="600")
        plt.tight_layout(pad=0.6)
        st.pyplot(fig,use_container_width=True); plt.close(fig)

    with cr:
        st.markdown('<div class="sec-header">📅 Atenciones por Año</div>',unsafe_allow_html=True)
        anios=[str(k) for k in ANA["por_anio"]]; counts=list(ANA["por_anio"].values())
        fig2,ax2=dark_fig(7,4.2)
        bc=["#29b6f6","#42a5f5","#5c6bc0"]
        bars2=ax2.bar(anios,counts,color=bc,edgecolor="none",width=0.5)
        ax2.spines[["top","right","left"]].set_visible(False)
        ax2.spines["bottom"].set_color("#3a4560")
        ax2.set_ylabel("Atenciones",color="#90a4ae",fontsize=9)
        ax2.tick_params(labelsize=10,labelcolor="#dce8f0")
        ax2.yaxis.grid(True,alpha=0.35); ax2.set_axisbelow(True)
        for bar,val in zip(bars2,counts):
            ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+300,
                     f"{val:,}",ha="center",fontsize=10,fontweight="700",color="#29b6f6")
        plt.tight_layout(pad=0.6)
        st.pyplot(fig2,use_container_width=True); plt.close(fig2)

    ca,cb=st.columns(2)

    with ca:
        st.markdown('<div class="sec-header">👥 Por Sexo</div>',unsafe_allow_html=True)
        fig3,ax3=plt.subplots(figsize=(4.5,4.5),facecolor="#1a1f2e")
        ax3.set_facecolor("#1a1f2e")
        sv=list(ANA["por_sexo"].values())
        w,t,at=ax3.pie(sv,labels=["Femenino","Masculino"],autopct="%1.1f%%",
            colors=["#f48fb1","#64b5f6"],startangle=90,
            wedgeprops={"edgecolor":"#1a1f2e","linewidth":3},pctdistance=0.73)
        for tx in t:   tx.set_color("#cfd8e3"); tx.set_fontsize(10); tx.set_fontweight("600")
        for at_ in at: at_.set_color("#12151e"); at_.set_fontsize(10); at_.set_fontweight("800")
        ax3.add_patch(plt.Circle((0,0),0.5,color="#1a1f2e"))
        plt.tight_layout(pad=0.3)
        st.pyplot(fig3,use_container_width=True); plt.close(fig3)

    with cb:
        st.markdown('<div class="sec-header">🧒 Curso de Vida</div>',unsafe_allow_html=True)
        cv=ANA["curso_vida"]
        fig4,ax4=plt.subplots(figsize=(4.5,4.5),facecolor="#1a1f2e")
        ax4.set_facecolor("#1a1f2e")
        w2,t2,at2=ax4.pie(list(cv.values()),labels=list(cv.keys()),autopct="%1.1f%%",
            colors=["#29b6f6","#ffca28","#66bb6a","#ce93d8"],startangle=120,
            wedgeprops={"edgecolor":"#1a1f2e","linewidth":3},pctdistance=0.73)
        for tx in t2:   tx.set_color("#cfd8e3"); tx.set_fontsize(9); tx.set_fontweight("600")
        for at_ in at2: at_.set_color("#12151e"); at_.set_fontsize(9); at_.set_fontweight("800")
        ax4.add_patch(plt.Circle((0,0),0.5,color="#1a1f2e"))
        plt.tight_layout(pad=0.3)
        st.pyplot(fig4,use_container_width=True); plt.close(fig4)

# ══════════════════════════════════════════════════════════
# ANALÍTICA
# ══════════════════════════════════════════════════════════
elif seccion == "📊 Analítica del Sistema":
    st.markdown('<div class="sec-header">📊 Análisis Exploratorio 2022–2024</div>',unsafe_allow_html=True)
    tab1,tab2,tab3=st.tabs(["🏥 Servicios y Especialidades","📍 Centros de Atención","💰 Valor de Servicios"])

    with tab1:
        c1,c2=st.columns([1.2,1])
        with c1:
            st.markdown("**Top 10 especialidades**")
            esp=ANA["top_esp"]; e_l=list(esp.keys()); e_v=list(esp.values())
            grad=["#29b6f6","#42a5f5","#5c85f5","#7986cb","#9575cd",
                  "#ba68c8","#ce93d8","#f06292","#ef9a9a","#ffb74d"]
            fig,ax=dark_fig(7,5)
            bars=ax.barh(e_l[::-1],e_v[::-1],color=grad[::-1],edgecolor="none",height=0.65)
            ax.spines[["top","right","left"]].set_visible(False)
            ax.spines["bottom"].set_color("#3a4560")
            ax.tick_params(axis="y",labelsize=8.5,labelcolor="#dce8f0")
            ax.tick_params(axis="x",labelcolor="#90a4ae")
            ax.set_xlabel("Atenciones",color="#90a4ae",fontsize=9)
            ax.set_xlim(0,max(e_v)*1.22)
            ax.xaxis.grid(True,alpha=0.35); ax.set_axisbelow(True)
            for bar,val in zip(bars,e_v[::-1]):
                ax.text(bar.get_width()+max(e_v)*0.012,bar.get_y()+bar.get_height()/2,
                        f"{val:,}",va="center",fontsize=8,color="#dce8f0",fontweight="600")
            plt.tight_layout(pad=0.6)
            st.pyplot(fig,use_container_width=True); plt.close(fig)

        with c2:
            st.markdown("**Distribución porcentual de servicios**")
            sd=ANA["servicios_dist"]
            fig2,ax2=plt.subplots(figsize=(5,5),facecolor="#1a1f2e")
            ax2.set_facecolor("#1a1f2e")
            sc=[SMETA[k]["color"] for k in sd]
            sl=[f"{SMETA[k]['icon']} {SMETA[k]['label']}" for k in sd]
            w,t,at=ax2.pie(list(sd.values()),labels=sl,autopct="%1.1f%%",
                colors=sc,startangle=140,
                wedgeprops={"edgecolor":"#1a1f2e","linewidth":3},pctdistance=0.76)
            for tx in t:   tx.set_color("#cfd8e3"); tx.set_fontsize(7.5)
            for at_ in at: at_.set_color("#12151e"); at_.set_fontsize(8); at_.set_fontweight("800")
            ax2.add_patch(plt.Circle((0,0),0.5,color="#1a1f2e"))
            plt.tight_layout(pad=0.3)
            st.pyplot(fig2,use_container_width=True); plt.close(fig2)

        st.markdown("**Resumen estadístico por tipo de servicio**")
        total=sum(sd.values())
        rows=[{"Servicio":f"{SMETA[k]['icon']} {SMETA[k]['label']}",
               "Atenciones":f"{cnt:,}",
               "Participación":f"{cnt/total*100:.1f}%",
               "Valor prom. COP":f"${ANA['valor_prom'][k]:,.0f}"}
              for k,cnt in sd.items()]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    with tab2:
        st.markdown("**Top 10 centros de atención**")
        tl=ANA["top_lug"]; t_l=list(tl.keys()); t_v=list(tl.values())
        tc=["#29b6f6","#42a5f5","#5c85f5","#7986cb","#9575cd",
            "#ba68c8","#ce93d8","#f06292","#ef9a9a","#ffb74d"]
        fig,ax=dark_fig(9,5)
        bars=ax.bar(t_l,t_v,color=tc,edgecolor="none",width=0.65)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#3a4560")
        ax.set_ylabel("Atenciones",color="#90a4ae",fontsize=9)
        ax.tick_params(axis="x",labelcolor="#dce8f0",labelsize=8)
        ax.tick_params(axis="y",labelcolor="#90a4ae")
        ax.yaxis.grid(True,alpha=0.35); ax.set_axisbelow(True)
        plt.xticks(rotation=32,ha="right")
        for bar,val in zip(bars,t_v):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+80,
                    f"{val:,}",ha="center",fontsize=8,fontweight="700",color="#dce8f0")
        plt.tight_layout(pad=0.8)
        st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.info(f"🏥 Los 10 centros principales concentran el **{sum(t_v)/ANA['total']*100:.1f}%** de las atenciones. UIMIST lidera con **{tl['UIMIST']:,}** atenciones.")

    with tab3:
        st.markdown("**Valor promedio de atención por tipo de servicio (COP)**")
        vp=ANA["valor_prom"]; v_l=[SMETA[k]["label"] for k in vp]; v_v=list(vp.values()); v_c=[SMETA[k]["color"] for k in vp]
        fig,ax=dark_fig(9,4.5)
        bars=ax.bar(v_l,v_v,color=v_c,edgecolor="none",width=0.55)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#3a4560")
        ax.set_ylabel("Valor promedio (COP)",color="#90a4ae",fontsize=9)
        ax.tick_params(axis="x",labelcolor="#dce8f0",labelsize=9,rotation=15)
        ax.tick_params(axis="y",labelcolor="#90a4ae")
        ax.yaxis.grid(True,alpha=0.35); ax.set_axisbelow(True)
        for bar,val in zip(bars,v_v):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+800,
                    f"${val:,.0f}",ha="center",fontsize=8.5,fontweight="700",color="#dce8f0")
        plt.tight_layout(pad=0.8)
        st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.warning("💡 **Insight:** Las Imágenes Diagnósticas tienen el mayor costo promedio ($123,000 COP), mientras que Vacunación es casi gratuita ($23 COP), reflejando su carácter preventivo masivo.")

# ══════════════════════════════════════════════════════════
# PREDICCIÓN
# ══════════════════════════════════════════════════════════
elif seccion == "🔮 Predicción de Servicio":
    st.markdown('<div class="sec-header">🔮 Predictor de Servicio con IA</div>',unsafe_allow_html=True)
    st.markdown("Ingresa los datos del **paciente migrante** y el sistema predecirá automáticamente el **tipo de servicio de salud** requerido, apoyando la planificación y asignación eficiente de recursos.")

    try:
        pipe,le=load_model()
    except Exception as e:
        st.error("No se pudo cargar el modelo de predicción.")
        st.info("Verifica que el archivo de Google Drive esté compartido como 'Cualquier persona con el enlace puede ver' y que el ID del archivo sea correcto.")
        st.exception(e)
        st.stop()

    with st.form("pred_form"):
        st.markdown("#### 👤 Datos del Paciente y Contexto de Atención")
        c1,c2=st.columns(2)
        with c1:
            sexo         =st.selectbox("🚻 Sexo del paciente",           SEXO_OPTS)
            curso_vida   =st.selectbox("🎂 Grupo de edad (Curso de Vida)",CURSO_VIDA_OPTS)
            especialidad =st.selectbox("🩺 Especialidad requerida",       ESPECIALIDAD_OPTS)
        with c2:
            lugar   =st.selectbox("📍 Centro de atención",LUGAR_OPTS)
            anio    =st.selectbox("📅 Año de atención",   ANIO_OPTS,index=2)
            mes_num =st.selectbox("🗓️ Mes de atención",   options=list(MESES.keys()),
                                  format_func=lambda x:MESES[x],index=0)
        submitted=st.form_submit_button("🔮 Predecir Servicio",type="primary",use_container_width=True)

    if submitted:
        with st.spinner("Analizando con IA…"):
            inp=pd.DataFrame([{"AÑO":anio,"MES_NUM":mes_num,"Sexo":sexo,
                                "Curso de vida":curso_vida,"LUGAR":lugar,"ESPECIALIDAD":especialidad}])
            pred_enc  =pipe.predict(inp)[0]
            pred_label=le.inverse_transform([pred_enc])[0]
            meta      =SMETA[pred_label]
            probas    =pipe.predict_proba(inp)[0]
            prob_dict =dict(zip(le.classes_,probas))
            top_probs =sorted(prob_dict.items(),key=lambda x:x[1],reverse=True)
            conf      =prob_dict[pred_label]*100

        st.markdown(f"""
        <div class="pred-box" style="background:{meta['bg']};border-color:{meta['color']};">
          <div class="pred-icon">{meta['icon']}</div>
          <div class="pred-service" style="color:{meta['color']};">{meta['label']}</div>
          <div class="pred-conf" style="color:{meta['color']};">Confianza del modelo: <strong>{conf:.1f}%</strong></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        cp,cr=st.columns([1.3,1])

        with cp:
            st.markdown("#### 📊 Probabilidad por tipo de servicio")
            for cls,prob in top_probs:
                m=SMETA[cls]; pct=prob*100
                st.markdown(f"""
                <div class="prob-row">
                  <div class="prob-label">{m['icon']} {m['label']} — {pct:.1f}%</div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{m['color']};"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        with cr:
            st.markdown("#### 🗒️ Resumen")
            st.markdown(f"""
            <table class="resumen-table">
              <tr><th>Campo</th><th>Valor</th></tr>
              <tr><td>🚻 Sexo</td><td>{sexo}</td></tr>
              <tr><td>🎂 Grupo etario</td><td>{curso_vida}</td></tr>
              <tr><td>🩺 Especialidad</td><td>{especialidad.title()}</td></tr>
              <tr><td>📍 Centro</td><td>{lugar}</td></tr>
              <tr><td>📅 Período</td><td>{MESES[mes_num]} {anio}</td></tr>
            </table>""", unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            val_est=ANA["valor_prom"].get(pred_label,0)
            st.metric("💰 Valor estimado",f"${val_est:,.0f} COP")
            if anio > 2024:
                st.caption("Nota: predicción futura estimada con un modelo entrenado con datos 2022-2024; úsala como referencia de planificación, no como valor observado.")

        interp={
            "CONSULTA":   "👨‍⚕️ Se anticipa una <strong>consulta médica</strong> general o especializada. Planificar disponibilidad de médicos y turnos de agenda.",
            "VACUNACION": "💉 Se proyecta una atención de <strong>vacunación</strong>. Verificar stock de vacunas y condiciones de cadena de frío.",
            "LABORATORIO":"🔬 El paciente requerirá <strong>pruebas de laboratorio</strong>. Coordinar insumos y personal de análisis.",
            "IMAGEN":     "🩻 Se requiere <strong>imagen diagnóstica</strong> (ecografía, radiología). Confirmar disponibilidad de equipos.",
            "TERAPIA":    "🧘 Se prevé una sesión de <strong>terapia</strong>. Asignar terapeuta especializado al turno.",
            "OTROS":      "🏥 Servicio de carácter <strong>diverso o complementario</strong>. Revisar necesidades específicas del caso.",
        }
        st.markdown(f'<div class="rec-box">💡 <strong>Recomendación operativa:</strong><br>{interp[pred_label]}</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ACERCA DEL PROYECTO
# ══════════════════════════════════════════════════════════
elif seccion == "📋 Acerca del Proyecto":
    st.markdown('<div class="sec-header">📋 Descripción del Proyecto</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.info("**📂 Dataset**\nAtenciones de salud para **población migrante en Bogotá** 2022–2024.\n- 47,315 registros · 11 variables · 27 centros")
        st.markdown("**Variables del modelo:**\n- `AÑO` / `MES` — Temporales\n- `ESPECIALIDAD` — Especialidad médica\n- `LUGAR` — Centro de atención\n- `Curso de vida` — Grupo etario\n- `Sexo` — Género del paciente")
    with c2:
        st.success("**🎯 Objetivo de Negocio**\nAnticipación de demanda de servicios de salud para población migrante, apoyando la **planificación estratégica y asignación eficiente de recursos**.")
        st.warning("**🤖 Modelo**\nRandom Forest Classifier — 300 árboles, balanceo de clases\n**Accuracy: 79% · F1-macro: 0.74**")
    st.markdown('<div class="sec-header">🌍 Alineación con ODS</div>',unsafe_allow_html=True)
    ods=[
        ("#2e7d32","🟢","ODS 3 – Salud y Bienestar","Garantizar atención oportuna para la población migrante promoviendo el bienestar sin discriminación."),
        ("#6a1b9a","🟣","ODS 10 – Reducción de Desigualdades","Reducir brechas en el acceso a la salud mediante planificación basada en datos y IA."),
        ("#e65100","🟠","ODS 11 – Ciudades Sostenibles","Contribuir a ciudades inclusivas optimizando los recursos sanitarios en Bogotá."),
    ]
    for col,(bdr,ico,tit,desc) in zip(st.columns(3),ods):
        col.markdown(f'<div class="kpi-card" style="border-top-color:{bdr};padding:1.3rem 1rem;"><div style="font-size:2rem;">{ico}</div><div style="font-weight:800;color:{bdr};margin-top:0.5rem;font-size:0.9rem;">{tit}</div><div style="font-size:0.82rem;color:rgba(255,255,255,0.65);margin-top:0.5rem;">{desc}</div></div>', unsafe_allow_html=True)
        
