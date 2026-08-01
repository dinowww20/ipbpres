import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import re
from collections import Counter
from scipy.stats import chi2_contingency

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    WC_OK = True
except ImportError:
    WC_OK = False

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Dashboard IPB Prestasi", layout="wide",
                    initial_sidebar_state="expanded")

with st.sidebar:
    st.markdown("### IPB Prestasi")
    theme_mode = st.radio("Mode tampilan:", ["Light", "Dark"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

if theme_mode == "Dark":
    bg_app, bg_panel = "#0F172A", "#1E293B"
    text_main, text_muted = "#F8FAFC", "#94A3B8"
    border_col, hover_bg = "#334155", "#0C2D25"
    chart_template = "plotly_dark"
else:
    bg_app, bg_panel = "#F8FAFC", "#FFFFFF"
    text_main, text_muted = "#1E293B", "#64748B"
    border_col, hover_bg = "#E2E8F0", "#ECFDF5"
    chart_template = "plotly_white"

ACCENT = "#0F6E56"       # teal — identitas IPB Prestasi
ACCENT2 = "#D85A30"      # coral — pasangan kontras (lambat/tidak/negatif)
ACCENT3 = "#EDA100"      # amber — netral/perhatian

# ═══════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp, p, h1, h2, h3, h4, h5, h6, label, li, .stMarkdown div {{
    font-family: 'Inter', -apple-system, sans-serif !important;
}}
.stApp {{ background: {bg_app} !important; color: {text_main} !important; }}
.main .block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}
[data-testid="stSidebar"] {{ background-color: {bg_panel} !important; border-right: 1px solid {border_col} !important; }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {{ color: {text_main} !important; font-size: 14px; }}
[data-testid="stSidebar"] .stSelectbox>label, [data-testid="stSidebar"] .stMultiSelect>label {{
    color: {text_muted} !important; font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: .6px;
}}
div[data-baseweb="select"] > div {{ background-color: {bg_panel} !important; color: {text_main} !important; border-color: {border_col} !important; }}
ul[role="listbox"] {{ background-color: {bg_panel} !important; }}
li[role="option"] {{ color: {text_main} !important; background-color: {bg_panel} !important; }}
[data-testid="stDataFrame"] {{ background-color: {bg_panel} !important; border: 1px solid {border_col}; border-radius: 8px; }}
.stTabs [data-baseweb="tab-list"] {{ background-color: {bg_panel} !important; border-radius: 14px; padding: 6px; gap: 4px; border: 1px solid {border_col} !important; }}
.stTabs button[role="tab"] {{ background: transparent !important; border-radius: 10px; padding: 10px 14px; font-weight: 600; font-size: 13px !important; }}
.stTabs button[role="tab"] p {{ color: {text_muted} !important; }}
.stTabs button[aria-selected="true"] {{ background: {ACCENT} !important; }}
.stTabs button[aria-selected="true"] p {{ color: white !important; }}
.kpi-card {{ background: {bg_panel}; border: 1px solid {border_col}; border-radius: 14px; padding: 18px 20px; }}
.kpi-label {{ color: {text_muted}; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .6px; margin-bottom: 6px; }}
.kpi-value {{ color: {text_main}; font-size: 30px; font-weight: 800; line-height: 1.1; }}
.kpi-sub {{ color: {text_muted}; font-size: 12.5px; margin-top: 4px; }}
.insight-box {{ background: {hover_bg}; border-left: 4px solid {ACCENT}; border-radius: 8px; padding: 14px 18px; margin: 10px 0 18px 0; }}
.insight-box p {{ color: {text_main} !important; font-size: 14px; margin: 0; line-height: 1.65; }}
.section-title {{ font-size: 20px; font-weight: 700; color: {text_main}; margin: 28px 0 4px 0; padding-bottom: 8px; border-bottom: 2px solid {border_col}; }}
.small-note {{ color: {text_muted}; font-size: 12px; font-style: italic; }}
</style>
""", unsafe_allow_html=True)


def sh(title):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def ib(text):
    st.markdown(f"<div class='insight-box'><p>{text}</p></div>", unsafe_allow_html=True)


def kpi(label, value, sub=""):
    st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>{label}</div>
    <div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>""",
                unsafe_allow_html=True)


def style_fig(fig, title=None, h=420, legend_below=False):
    fig.update_layout(
        template=chart_template, height=h,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color=text_main, size=12.5),
        title=dict(text=title, font=dict(size=15, weight=700)) if title else None,
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5) if legend_below else {},
    )
    fig.update_xaxes(gridcolor=border_col, zerolinecolor=border_col)
    fig.update_yaxes(gridcolor=border_col, zerolinecolor=border_col)
    return fig


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════
def find_col(df, keywords, exclude=None):
    for kw in keywords:
        matches = [c for c in df.columns if kw.lower() in c.lower()]
        if exclude:
            matches = [m for m in matches if not any(e.lower() in m.lower() for e in exclude)]
        if matches:
            return matches[0]
    return None


@st.cache_data
def load_data():
    candidates = glob.glob("data/*.xlsx") + glob.glob("*.xlsx") + glob.glob("data/**/*.xlsx", recursive=True)
    if not candidates:
        return None, {}
    raw = pd.read_excel(candidates[0], sheet_name="Form Responses 1")

    cmap = {}
    cmap['gender'] = find_col(raw, ['Jenis Kelamin'])
    cmap['angkatan'] = find_col(raw, ['Angkatan'])
    cmap['tempat_tinggal'] = find_col(raw, ['Tempat Tinggal'])
    cmap['jalur_masuk'] = find_col(raw, ['Jalur Masuk'])
    cmap['komunitas'] = find_col(raw, ['komunitas yang pernah'])
    cmap['adaptasi_cepat'] = find_col(raw, ['cepat beradaptasi dengan lingkungan'])
    cmap['alasan_cepat'] = find_col(raw, ['merasa cepat beradaptasi'])
    cmap['alasan_lambat'] = find_col(raw, ['tidak cepat beradaptasi'])
    cmap['kerja_sendiri'] = find_col(raw, ['bekerja sendirian dibandingkan'])
    cmap['alasan_sendiri'] = find_col(raw, ['memilih untuk bekerja sendirian'])
    cmap['alasan_tim'] = find_col(raw, ['memilih untuk bekerja sama'])
    cmap['profil_mhs'] = find_col(raw, ['Selama berkuliah manakah'])
    cmap['minat_lomba'] = find_col(raw, ['minat untuk mengikuti lomba'])
    cmap['alasan_minat'] = find_col(raw, ['minat untuk berkompetisi'])
    cmap['alasan_tidak_minat'] = find_col(raw, ['tidak memiliki minat berkompetisi'])
    cmap['bidang_minat_utama'] = find_col(raw, ['paling mencerminkan minat'])

    bidang_map = {
        'debat': dict(kompetisi='pernah mengikuti kompetisi debat', skill='pembinaan pada bida',
                      fasilitas='mendukung perkembangan kamu'),
    }
    idx_bidang = {
        'Debat & Diplomasi': dict(kompetisi=44, skill=47, fasilitas=49),
        'ICT–Robotika': dict(kompetisi=51, skill=54, fasilitas=56),
        'Pengabdian': dict(kompetisi=58, skill=62, fasilitas=64),
        'Sains & Penalaran': dict(kompetisi=66, skill=69, fasilitas=71),
        'Bisnis': dict(kompetisi=73, skill=76, fasilitas=78),
    }
    for bidang, idx in idx_bidang.items():
        for key, i in idx.items():
            cmap[f'{bidang}__{key}'] = raw.columns[i] if i < len(raw.columns) else None

    cmap['metode_pembinaan'] = find_col(raw, ['paling sesuai untuk pelaksanaan pembinaan'])
    cmap['ukuran_kelompok'] = find_col(raw, ['nyaman dalam pembinaan yang dilakukan'])
    cmap['aktivitas_pembinaan'] = find_col(raw, ['aktivitas pembinaan yang paling sesuai'])
    cmap['struktur_program'] = find_col(raw, ['program pembinaan kompetisi. program'])
    cmap['frekuensi_pembinaan'] = find_col(raw, ['pelaksanaan pembinaan yang ideal'])
    cmap['tipe_mentor'] = find_col(raw, ['tipe mentor/pembimbing yang anda harapkan'])
    cmap['kriteria_mentor'] = find_col(raw, ['kriteria mentor yang'])

    cmap['pengetahuan_fasilitasi'] = find_col(raw, ['pernah sekadar mengetahui salah satu'])
    cmap['program_diketahui'] = find_col(raw, ['program fasilitasi kompetitif dari ipb prestasi yang anda tahu'])
    cmap['sumber_info'] = find_col(raw, ['dari mana anda mengetahui adanya fasilitasi kompetitif'])
    cmap['promosi_efektif'] = find_col(raw, ['seberapa efektif promosi'])
    cmap['media_suka'] = find_col(raw, ['media promosi dan informasi'])
    cmap['pernah_pakai'] = find_col(raw, ['pernah menggunakan fasilitasi kompetitif'])
    cmap['kualitas_layanan'] = find_col(raw, ['kualitas layanan saat'])
    cmap['kemudahan_alur'] = find_col(raw, ['kemudahan mekanisme'])
    cmap['kecepatan_verif'] = find_col(raw, ['kecepatan verifikasi'])
    cmap['kendala'] = find_col(raw, ['kendala utama dalam mengakses'])
    cmap['evaluasi'] = find_col(raw, ['apa evaluasi untuk'])
    cmap['saran'] = find_col(raw, ['saran yang ingin anda berikan'])

    q_keywords = [
        'menganalisis data dan memecahkan masalah kompleks',
        'menulis paper ilmiah atau presentasi riset',
        'studi kasus bisnis atau teknologi terbaru',
        'coding, ai, atau cybersecurity',
        'mengembangkan ide bisnis dan strategi pemasaran',
        'pitching ide ke investor atau membuat business plan',
        'menyusun argumen yang solid dan persuasif',
        'diskusi isu sosial, politik, atau ekonomi',
    ]
    q_labels = ['Q1_analisis_data', 'Q2_paper_ilmiah', 'Q3_bisnis_teknologi', 'Q4_coding_AI',
                'Q5_ide_bisnis', 'Q6_pitching', 'Q7_argumen_persuasif', 'Q8_diskusi_sosial']
    for lbl, kw in zip(q_labels, q_keywords):
        matches = [c for c in raw.columns if kw in c.lower() and c.strip().endswith('2')]
        cmap[lbl] = matches[0] if matches else None

    rename_dict = {v: k for k, v in cmap.items() if v is not None}
    df = raw.rename(columns=rename_dict)
    df = df.loc[:, ~df.columns.duplicated(keep='last')]

    return df, cmap


df, cmap = load_data()

if df is None:
    st.error("File data (.xlsx) tidak ditemukan. Pastikan file diletakkan di folder `data/` "
             "atau di root repo, dengan sheet bernama **'Form Responses 1'**.")
    st.stop()

missing = [k for k, v in cmap.items() if v is None]
with st.sidebar:
    st.markdown("---")
    with st.expander(f"Debug: kolom terdeteksi ({len(cmap)-len(missing)}/{len(cmap)})"):
        for k, v in cmap.items():
            st.caption(f"{'✅' if v else '❌'} `{k}`" + (f" → {v[:45]}..." if v else " → TIDAK DITEMUKAN"))

LABEL_ALASAN_CEPAT = {
    'Fleksibel dan Mudah Belajar (Dapat menyesuaikan perilaku dengan situasi dan cepat mempelajari aturan, budaya dan kebiasaan baru)': 'Fleksibel & mudah belajar',
    'Percaya Diri (Mudah menerima hal baru tanpa prasangka dan melihat perubahan sebagai peluang bukan ancaman)': 'Percaya diri',
    'Komunikatif (Aktif berkomunikasi dan mudah bergaul serta berani memulai percakapan)': 'Komunikatif aktif',
    'Tidak bergantung pada orang lain dan dapat mengelola emosi dengan baik saat menghadapi kesulitan': 'Tak bergantung orang lain',
    'Terbiasa dengan situasi baru (sering pindah) dan tidak takut keluar dari zona nyaman': 'Terbiasa situasi baru',
}
LABEL_ALASAN_LAMBAT = {
    'Sifat Pemalu dan Rasa Canggung (Merasa tidak nyaman menjadi pusat perhatian dan butuh waktu lama untuk membuka diri)': 'Pemalu & canggung',
    'Takut akan penolakan dan Salah Paham (Khawatir perkataan atau tindakan akan disalahartikan, takut tidak diterima dan sangat hati-hati)': 'Takut ditolak/salah paham',
    'Kesulitan membaca sistuasi sosial (Bingung menentukan topik pembicaraan yang tepat, merasa seperti orang luar karena tidak mengerti dinamika kelompok)': 'Sulit baca situasi sosial',
    'Keterbatasan pengalaman dengan perubahan (Tidak terbiasa berada di lingkungan yang benar-benar baru dan kurangnya referensi untuk menghadapi situasi seperti ini)': 'Kurang pengalaman perubahan',
    'Sangat terbiasa dengan cara cara lama dan sulit menerima sistem, aturan atau budaya yang berbeda': 'Terbiasa cara lama',
}
LABEL_KOMUNITAS = {
    'Tidak satupun dari komunitas di atas pernah/sedang saya ikut.': 'Tidak ikut komunitas',
}
LABEL_PROFIL = {
    'The Balanced Leader: Mengintegrasikan Akademik dan Organisasi (Aktif berorganisasi dan tetap menjaga prestasi akademik dengan baik)': 'The Balanced Leader',
    'The Focused Enthusiast: Meraih Prestasi di Dua Dunia (Fokus pada akademik, namun tetap menekuni hobi/minat secara serius hingga level kompetitif)': 'The Focused Enthusiast',
    'The Academic Specialist: Fokus Mendalam pada Prestasi Akademik (Menjadikan Akademik sebagai prioritas utama dan aren pengembangan diri yang paling bermakna)': 'The Academic Specialist',
    'Belum tahu yang mana': 'Belum tahu profilnya',
    'The Passionate Competitor: Berprestasi di Jalur Minat (Menekuni hobi atau minat khusus dengan serius hingga level kompetitif)': 'The Passionate Competitor',
}
BIDANG_LIST = ['Debat & Diplomasi', 'ICT–Robotika', 'Pengabdian', 'Sains & Penalaran', 'Bisnis']
BIDANG_MAP_RAW = {
    'Debat & Diplomasi': 'Debat dan Diplomasi', 'ICT–Robotika': 'ICT - Robotika',
    'Pengabdian': 'Pengabdian', 'Sains & Penalaran': 'Sains dan Penalaran', 'Bisnis': 'Bisnis',
}
COLORWAY = ['#0F6E56', '#378ADD', '#D85A30', '#EDA100', '#7F77DD', '#D4537E']

# ═══════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("---")
    st.markdown("### Filter data")
    f_gender = st.multiselect("Jenis kelamin", options=sorted(df['gender'].dropna().unique()) if 'gender' in df else [])
    f_jalur = st.multiselect("Jalur masuk", options=sorted(df['jalur_masuk'].dropna().unique()) if 'jalur_masuk' in df else [])
    f_bidang = st.multiselect("Bidang minat utama", options=BIDANG_LIST)

df_f = df.copy()
if f_gender and 'gender' in df_f:
    df_f = df_f[df_f['gender'].isin(f_gender)]
if f_jalur and 'jalur_masuk' in df_f:
    df_f = df_f[df_f['jalur_masuk'].isin(f_jalur)]
if f_bidang and 'bidang_minat_utama' in df_f:
    raw_vals = [BIDANG_MAP_RAW[b] for b in f_bidang]
    df_f = df_f[df_f['bidang_minat_utama'].isin(raw_vals)]

N = len(df_f)

# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
st.markdown(f"""<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
<div><span style='font-size:26px; font-weight:800; color:{text_main}'>Dashboard Analisis Survey Prestasi</span><br>
<span style='color:{text_muted}; font-size:13.5px'>IPB University — Fasilitasi Kompetitif Mahasiswa</span></div>
<div style='text-align:right; color:{text_muted}; font-size:12.5px'>n = {N} responden{" (terfilter)" if N != len(df) else f" dari {len(df)}"}</div>
</div>""", unsafe_allow_html=True)

t1, t2, t3, t4, t5, t6, t7 = st.tabs(
    ["Overview", "Demografi", "Karakteristik", "Peminatan Prestasi",
     "Pola Pembinaan", "Fasilitasi Kompetisi", "Insight Teks"])

# ═══════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════
with t1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pct_cepat = (df_f['adaptasi_cepat'] == 'Ya').mean() * 100 if 'adaptasi_cepat' in df_f else None
        kpi("Adaptasi cepat", f"{pct_cepat:.0f}%" if pct_cepat is not None else "–",
            f"dari {df_f['adaptasi_cepat'].notna().sum()} yang menjawab" if 'adaptasi_cepat' in df_f else "")
    with c2:
        pct_minat = (df_f['minat_lomba'] == 'Ya').mean() * 100 if 'minat_lomba' in df_f else None
        kpi("Berminat kompetisi", f"{pct_minat:.0f}%" if pct_minat is not None else "–", "dari total responden")
    with c3:
        pct_tim = (df_f['kerja_sendiri'] == 'Tidak').mean() * 100 if 'kerja_sendiri' in df_f else None
        kpi("Prefer kerja tim", f"{pct_tim:.0f}%" if pct_tim is not None else "–", "vs kerja sendiri")
    with c4:
        pct_tahu = (df_f['pengetahuan_fasilitasi'] == 'Ya').mean() * 100 if 'pengetahuan_fasilitasi' in df_f else None
        kpi("Tahu Fasilitasi Kompetitif", f"{pct_tahu:.0f}%" if pct_tahu is not None else "–", "minimal 1 program")

    st.markdown("<br>", unsafe_allow_html=True)
    sh("Profil Mahasiswa Selama Perkuliahan")
    if 'profil_mhs' in df_f:
        pc = df_f['profil_mhs'].map(LABEL_PROFIL).fillna(df_f['profil_mhs']).value_counts().sort_values(ascending=True)
        fig = px.bar(pc, x=pc.values, y=pc.index, orientation='h', text=pc.values,
                     color_discrete_sequence=[ACCENT])
        fig.update_traces(textposition='outside')
        st.plotly_chart(style_fig(fig, h=380), use_container_width=True)
        top = pc.idxmax()
        ib(f"Profil dominan adalah <b>{top}</b> ({pc.max()} orang, {pc.max()/pc.sum()*100:.0f}%) — "
           f"mayoritas mahasiswa mengejar keseimbangan antara akademik dan aktivitas non-akademik, bukan fokus tunggal.")

    sh("Sebaran Bidang Minat Utama")
    if 'bidang_minat_utama' in df_f:
        bc = df_f['bidang_minat_utama'].value_counts()
        bc = bc[bc.index.isin(BIDANG_MAP_RAW.values())]
        fig = px.pie(values=bc.values, names=bc.index, hole=0.5, color_discrete_sequence=COLORWAY)
        fig.update_traces(textinfo='label+percent')
        st.plotly_chart(style_fig(fig, h=400), use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — DEMOGRAFI
# ═══════════════════════════════════════════════════════
with t2:
    d1, d2 = st.columns(2)
    with d1:
        sh("Jenis Kelamin")
        if 'gender' in df_f:
            gc = df_f['gender'].value_counts()
            fig = px.pie(values=gc.values, names=gc.index, hole=0.5,
                         color_discrete_sequence=['#378ADD', '#85B7EB'])
            fig.update_traces(textinfo='label+percent+value')
            st.plotly_chart(style_fig(fig, h=350), use_container_width=True)
    with d2:
        sh("Tempat Tinggal")
        if 'tempat_tinggal' in df_f:
            tc = df_f['tempat_tinggal'].value_counts()
            fig = px.pie(values=tc.values, names=tc.index, hole=0.5,
                         color_discrete_sequence=['#0F6E56', '#5DCAA5', '#A8E0CB'])
            fig.update_traces(textinfo='label+percent+value')
            st.plotly_chart(style_fig(fig, h=350), use_container_width=True)

    d3, d4 = st.columns(2)
    with d3:
        sh("Jalur Masuk")
        if 'jalur_masuk' in df_f:
            jc = df_f['jalur_masuk'].value_counts().sort_values(ascending=True)
            fig = px.bar(jc, x=jc.values, y=jc.index, orientation='h', text=jc.values,
                         color=jc.values, color_continuous_scale='Blues')
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, h=350), use_container_width=True)
    with d4:
        sh("Distribusi Angkatan")
        if 'angkatan' in df_f:
            ac = df_f['angkatan'].value_counts().sort_index(ascending=False)
            fig = px.bar(x=ac.index.astype(str), y=ac.values, text=ac.values,
                         color_discrete_sequence=[ACCENT])
            fig.update_traces(textposition='outside')
            st.plotly_chart(style_fig(fig, h=350), use_container_width=True)

    sh("Komunitas yang Pernah/Sedang Diikuti")
    if 'komunitas' in df_f:
        ks = df_f['komunitas'].dropna().str.split(',').explode().str.strip()
        ks = ks.map(lambda x: LABEL_KOMUNITAS.get(x, x))
        kc = ks.value_counts().sort_values(ascending=True)
        fig = px.bar(kc, x=kc.values, y=kc.index, orientation='h', text=kc.values,
                     color=kc.values, color_continuous_scale='Greens')
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, h=380), use_container_width=True)
        ib("Multi-select — jumlah bisa melebihi n karena satu responden bisa mengikuti lebih dari satu komunitas.")

# ═══════════════════════════════════════════════════════
# TAB 3 — KARAKTERISTIK
# ═══════════════════════════════════════════════════════
def tornado_chart(df_in, ya_col, alasan_ya_col, alasan_tidak_col, label_ya, label_tidak,
                   map_ya, map_tidak, color_ya=ACCENT, color_tidak=ACCENT2, title=""):
    ya_valid = df_in[(df_in[ya_col] == 'Ya') & (df_in[alasan_ya_col].isin(map_ya.keys()))]
    ya_c = ya_valid[alasan_ya_col].map(map_ya).value_counts()
    tidak_valid = df_in[(df_in[ya_col] == 'Tidak') & (df_in[alasan_tidak_col].isin(map_tidak.keys()))]
    tidak_c = tidak_valid[alasan_tidak_col].map(map_tidak).value_counts()

    if ya_c.empty or tidak_c.empty:
        st.info("Data tidak cukup untuk kombinasi filter ini.")
        return None, None

    ya_c = ya_c.sort_values(ascending=True)
    tidak_c = tidak_c.sort_values(ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(y=tidak_c.index, x=-tidak_c.values, orientation='h',
                          marker_color=color_tidak, name=label_tidak,
                          text=tidak_c.values, textposition='outside'))
    fig.add_trace(go.Bar(y=ya_c.index, x=ya_c.values, orientation='h',
                          marker_color=color_ya, name=label_ya,
                          text=ya_c.values, textposition='outside'))
    fig.update_layout(barmode='overlay', title=title)
    return fig, (ya_c, tidak_c)


with t3:
    sh("Kecepatan Adaptasi di Lingkungan Baru")
    if all(k in df_f for k in ['adaptasi_cepat', 'alasan_cepat', 'alasan_lambat']):
        n_ya = (df_f['adaptasi_cepat'] == 'Ya').sum()
        n_tidak = (df_f['adaptasi_cepat'] == 'Tidak').sum()
        fig, res = tornado_chart(df_f, 'adaptasi_cepat', 'alasan_cepat', 'alasan_lambat',
                                  f'Cepat (n={n_ya})', f'Lambat (n={n_tidak})',
                                  LABEL_ALASAN_CEPAT, LABEL_ALASAN_LAMBAT)
        if fig:
            st.plotly_chart(style_fig(fig, h=420, legend_below=True), use_container_width=True)
            ib(f"<b>{n_ya}</b> mahasiswa ({n_ya/(n_ya+n_tidak)*100:.0f}%) cepat beradaptasi, "
               f"didorong utamanya oleh fleksibilitas dan kemudahan belajar. Yang lambat beradaptasi "
               f"({n_tidak} orang) paling sering terkendala rasa malu/canggung, bukan karena masalah teknis.")

    sh("Preferensi Kerja: Sendiri vs Tim")
    if all(k in df_f for k in ['kerja_sendiri', 'alasan_sendiri', 'alasan_tim']):
        label_sendiri = {
            'Memiliki kendali penuh atas proses dan hasil (bebas menentukan metode, prioritas dan lagkah kerja sendiiri serta tidak perlu kompromi atau menunggu persetujuan orang lain)': 'Kendali penuh proses & hasil',
            'Memiliki kendali penuh atas proses dan hasil (bebas menentukan metode, prioritas dan langkah kerja sendiri serta tidak perlu kompromi atau menunggu persetujuan orang lain)': 'Kendali penuh proses & hasil',
            'Fokus dan Kualitas kerja lebih terjaga (bebas dari gangguan dan interupsi rekan kerja dan dapat berkonsentrasi penuh untuk menghasilkan hasil terbaik)': 'Fokus & kualitas kerja terjaga',
            'Lebih cepat dan efisien (Tidak perlu waktu untuk koordinasi, rapat dan diskusi dan menghindari prosedur kelompok yang berbelit)': 'Lebih cepat & efisien',
            'Cenderung intorvert dan merasa lebih produktif dalam kesendirian dan ketenangan': 'Introvert, produktif sendiri',
            'Pengalaman negatif dalam bekerja sama (Pernah mengalami konflik, ketidakcocokan atau ketergantungan pada rekan tim)': 'Pengalaman negatif kerja tim',
        }
        label_tim = {
            'Merasa lebih termotivasi dan bersemangat ketika bekerja dalam berkelompok, energi dan antusiasme rekan kerja dapat salaing menular secara positif': 'Termotivasi energi rekan',
            'Menikmati proses belajar dan bertukar pengetahuan (Dapat saling mengoreksi dan menyempurnakan pekerjaan bersama-sama)': 'Belajar & tukar pengetahuan',
            'Percaya pada kekuatan sinegri dan hasil yang lebih baik (Hasil akhir dirasa lebih kaya dan komprehensif karena ada banyak kontibutor)': 'Sinergi, hasil lebih baik',
            'Menghindari Rasa kesendirian dan isolasi (Kerjasama memberikan interaksi sosial yang menyenangkan dan mengurasi rasa jenuh)': 'Hindari kesendirian/isolasi',
            'Risiko dan tekanana tidak ditanggung sendirian, pekerjaan ringan dan cepat ketika dibagi': 'Risiko/beban dibagi',
        }
        n_s = (df_f['kerja_sendiri'] == 'Ya').sum()
        n_t = (df_f['kerja_sendiri'] == 'Tidak').sum()
        fig2, res2 = tornado_chart(df_f, 'kerja_sendiri', 'alasan_sendiri', 'alasan_tim',
                                    f'Sendiri (n={n_s})', f'Tim (n={n_t})',
                                    label_sendiri, label_tim, color_ya='#378ADD', color_tidak='#7F77DD')
        if fig2:
            st.plotly_chart(style_fig(fig2, h=420, legend_below=True), use_container_width=True)
            total = n_s + n_t
            ib(f"Preferensi nyaris seimbang: {n_s} orang ({n_s/total*100:.0f}%) sendiri vs {n_t} orang "
               f"({n_t/total*100:.0f}%) tim — bukan tren dominan ke satu arah. Ini penting untuk desain "
               f"program pembinaan: sediakan opsi kelompok kecil <i>dan</i> jalur individual.")

    sh("Minat Berkompetisi")
    if all(k in df_f for k in ['minat_lomba', 'alasan_minat']):
        label_minat = {
            'Merasakan Pengalaman seru dan Dinamika yang Menantang': 'Pengalaman & tantangan baru',
            'Mencapai Target Pribadi dan Membangun Rasa Percaya Diri': 'Target pribadi & percaya diri',
            'Nambahin CV dan Peluang ke Depannya': 'Nambah CV & peluang',
            'Sebagai Tolak Ukur dan Pembuktian Kemampuan Diri': 'Tolok ukur kemampuan diri',
            'Mendapatkan Apresiasi dan Pengakuan': 'Apresiasi & pengakuan',
            'Menjalin Koneksi dengan Sesama Peminat': 'Koneksi sesama peminat',
            'Pertemanan dengan mahasiswa yang memiliki passion yang sama': 'Pertemanan sepassion',
        }
        minat_valid = df_f[(df_f['minat_lomba'] == 'Ya') & (df_f['alasan_minat'].isin(label_minat.keys()))]
        minat_c = minat_valid['alasan_minat'].map(label_minat).value_counts().sort_values(ascending=True)
        n_minat = (df_f['minat_lomba'] == 'Ya').sum()
        n_tidak_minat = (df_f['minat_lomba'] == 'Tidak').sum()
        if not minat_c.empty:
            fig3 = px.bar(minat_c, x=minat_c.values, y=minat_c.index, orientation='h', text=minat_c.values,
                          color_discrete_sequence=[ACCENT])
            fig3.update_traces(textposition='outside')
            st.plotly_chart(style_fig(fig3, title=f"Alasan minat berkompetisi (n={n_minat})", h=380), use_container_width=True)
            ib(f"<b>{n_minat}</b> dari {n_minat+n_tidak_minat} responden ({n_minat/(n_minat+n_tidak_minat)*100:.0f}%) "
               f"berminat berkompetisi — dominan didorong pengalaman & tantangan baru, bukan sekadar CV. "
               f"<span class='small-note'>Kelompok tidak berminat n={n_tidak_minat}, terlalu kecil untuk tren statistik, treat sebagai catatan kualitatif.</span>")

# ═══════════════════════════════════════════════════════
# TAB 4 — PEMINATAN PRESTASI
# ═══════════════════════════════════════════════════════
with t4:
    sh("Proporsi Pernah Ikut Kompetisi per Bidang Minat")
    rows = []
    for b in BIDANG_LIST:
        col = cmap.get(f'{b}__kompetisi')
        if col and col in df.columns:
            s = df_f[col].dropna()
            n = len(s)
            ya = (s == 'Ya').sum()
            rows.append({'Bidang': b, 'n': n, 'Ya': ya, 'pct': ya / n * 100 if n > 0 else 0})
    if rows:
        bidang_df = pd.DataFrame(rows).sort_values('pct')
        fig = px.bar(bidang_df, x='pct', y='Bidang', orientation='h',
                     text=[f"{p:.0f}% ({y}/{n})" for p, y, n in zip(bidang_df['pct'], bidang_df['Ya'], bidang_df['n'])],
                     color_discrete_sequence=[ACCENT])
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_range=[0, 100])
        st.plotly_chart(style_fig(fig, h=350), use_container_width=True)
        small_n = bidang_df[bidang_df['n'] < 15]['Bidang'].tolist()
        if small_n:
            st.markdown(f"<span class='small-note'>Catatan: n kecil untuk {', '.join(small_n)} — interpretasikan sebagai indikasi awal.</span>", unsafe_allow_html=True)

    sh("Uji Asosiasi (Chi-square & Cramér's V)")
    st.markdown("<span class='small-note'>Sampel bersifat purposive (non-probability) — hasil dideskripsikan sebagai pola dalam sampel, bukan generalisasi populasi.</span>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        if all(k in df_f for k in ['jalur_masuk', 'adaptasi_cepat']):
            valid = df_f[df_f['jalur_masuk'].isin(['SNBP', 'SNBT', 'Mandiri', 'Jaketos', 'PIN/ Talenta IPB'])]
            ct = pd.crosstab(valid['jalur_masuk'], valid['adaptasi_cepat'])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, p, dof, _ = chi2_contingency(ct)
                nobs = ct.sum().sum()
                v = np.sqrt(chi2 / (nobs * (min(ct.shape) - 1)))
                ctp = ct.div(ct.sum(axis=1), axis=0) * 100
                ctp = ctp.loc[ctp.get('Ya', pd.Series(dtype=float)).sort_values(ascending=True).index] if 'Ya' in ctp else ctp
                fig = go.Figure()
                for cat, color in zip(['Tidak', 'Ya'], [ACCENT2, ACCENT]):
                    if cat in ctp.columns:
                        fig.add_trace(go.Bar(y=ctp.index, x=ctp[cat], orientation='h', name=cat,
                                              marker_color=color, text=[f"{v:.0f}%" for v in ctp[cat]], textposition='inside'))
                fig.update_layout(barmode='stack')
                st.plotly_chart(style_fig(fig, title=f"Jalur Masuk × Adaptasi (V={v:.2f}, p={p:.3f})", h=320, legend_below=True), use_container_width=True)

    with cb:
        if all(k in df_f for k in ['bidang_minat_utama', 'kerja_sendiri']):
            valid = df_f[df_f['bidang_minat_utama'].isin(BIDANG_MAP_RAW.values())]
            ct2 = pd.crosstab(valid['bidang_minat_utama'], valid['kerja_sendiri'])
            if ct2.shape[0] > 1 and ct2.shape[1] > 1:
                chi2b, pb, dofb, _ = chi2_contingency(ct2)
                nb = ct2.sum().sum()
                vb = np.sqrt(chi2b / (nb * (min(ct2.shape) - 1)))
                ct2p = ct2.div(ct2.sum(axis=1), axis=0) * 100
                fig2 = go.Figure()
                for cat, color, lbl in zip(['Tidak', 'Ya'], ['#7F77DD', '#378ADD'], ['Lebih suka tim', 'Lebih suka sendiri']):
                    if cat in ct2p.columns:
                        fig2.add_trace(go.Bar(y=ct2p.index, x=ct2p[cat], orientation='h', name=lbl,
                                               marker_color=color, text=[f"{v:.0f}%" for v in ct2p[cat]], textposition='inside'))
                fig2.update_layout(barmode='stack')
                st.plotly_chart(style_fig(fig2, title=f"Bidang × Gaya Kerja (V={vb:.2f}, p={pb:.3f})", h=320, legend_below=True), use_container_width=True)

    sh("Fasilitas Paling Diharapkan per Bidang")
    fas_rows = []
    for b in BIDANG_LIST:
        col = cmap.get(f'{b}__fasilitas')
        if col and col in df.columns:
            vc = df_f[col].value_counts()
            if len(vc) > 0:
                top = vc.index[0]
                fas_rows.append({'Bidang': b, 'Fasilitas paling diharapkan': top,
                                  '%': f"{vc.iloc[0]/vc.sum()*100:.0f}%", 'n': vc.sum()})
    if fas_rows:
        st.dataframe(pd.DataFrame(fas_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
# TAB 5 — POLA PEMBINAAN
# ═══════════════════════════════════════════════════════
with t5:
    sh("Preferensi Format Pembinaan")
    label_metode = {
        'Kelompok Kecil (Hanya 1 tim/ maksimal 5 orang)': 'Kelompok kecil',
        '1 on 1 atau Personal (hanya diri sendiri)': 'Personal (1-on-1)',
        'Kelompok Besar (Pembinaan kelas seperti enrichment dsb.)': 'Kelompok besar',
    }
    label_ukuran = {
        'Pendampingan dan Pembimbingan Personal (Mentoring/coaching dengan dosen/alumni/tokoh inspiratif)': 'Coaching personal',
        'Penguatan Relasi dan Keakraban(Sikrab/makrab, diskusi santai, dan studi kasus)': 'Relasi & keakraban',
        'Pelatihan Keterampilan dan Simulasi Lapangan (Simulasi lomba dan Praktik )': 'Simulasi & praktik',
        'Penguatan Pengetahuan dan Pemahaman Teoritis (Training dan workshop)': 'Teori (workshop)',
    }
    label_struktur = {
        'Pembinaan bebas (fleksibel) dan tidak terafiliasi dengan organisasi formal kampus, bisa diikuti tanpa komitmen keanggotaan atau struktur formal': 'Fleksibel',
        'Pembinaan terstruktur (terikat) dan diselenggarakan oleh pihak yang berafiliasi dengan organisasi formal kampus , dan biasanya memerlukan komitmen keikutsertaan': 'Terstruktur',
    }

    def pct_series(col, lmap):
        if col not in df_f.columns:
            return pd.Series(dtype=float)
        s = df_f[col].dropna().map(lambda x: lmap.get(x, x)).value_counts()
        return (s / s.sum() * 100).round(0) if s.sum() > 0 else s

    rows_pref = {
        'Ukuran kelompok': pct_series('metode_pembinaan', label_metode),
        'Aktivitas ideal': pct_series('ukuran_kelompok', label_ukuran),
        'Struktur pembinaan': pct_series('aktivitas_pembinaan', label_struktur),
    }
    fig = go.Figure()
    y_labels = list(rows_pref.keys())
    for row_name, series in rows_pref.items():
        left = 0
        for j, (cat, val) in enumerate(series.items()):
            fig.add_trace(go.Bar(y=[row_name], x=[val], orientation='h', name=cat,
                                  marker_color=COLORWAY[j % len(COLORWAY)],
                                  text=f"{cat} {val:.0f}%", textposition='inside',
                                  showlegend=False, base=left))
            left += val
    fig.update_layout(barmode='overlay', xaxis_range=[0, 100])
    st.plotly_chart(style_fig(fig, h=320), use_container_width=True)

    sh("Pola Preferensi Pembinaan per Bidang Minat")
    if 'bidang_minat_utama' in df_f and 'metode_pembinaan' in df_f:
        sub = df_f[df_f['bidang_minat_utama'].isin(BIDANG_MAP_RAW.values())].copy()
        sub['metode_short'] = sub['metode_pembinaan'].map(label_metode)
        sub['ukuran_short'] = sub['ukuran_kelompok'].map(label_ukuran)
        sub['struktur_short'] = sub['aktivitas_pembinaan'].map(label_struktur)

        heat_rows = []
        for b_raw in sub['bidang_minat_utama'].unique():
            g = sub[sub['bidang_minat_utama'] == b_raw]
            n = len(g)
            heat_rows.append({
                'Bidang': b_raw, 'n': n,
                '% Kelompok kecil': (g['metode_short'] == 'Kelompok kecil').sum() / g['metode_short'].notna().sum() * 100 if g['metode_short'].notna().sum() else np.nan,
                '% Coaching personal': (g['ukuran_short'] == 'Coaching personal').sum() / g['ukuran_short'].notna().sum() * 100 if g['ukuran_short'].notna().sum() else np.nan,
                '% Fleksibel': (g['struktur_short'] == 'Fleksibel').sum() / g['struktur_short'].notna().sum() * 100 if g['struktur_short'].notna().sum() else np.nan,
            })
        heat_df = pd.DataFrame(heat_rows).set_index('Bidang')
        n_col = heat_df.pop('n')
        heat_df = heat_df.round(0)

        fig_h = go.Figure(data=go.Heatmap(
            z=heat_df.values, x=heat_df.columns, y=[f"{i} (n={n_col[i]})" for i in heat_df.index],
            colorscale='Blues', text=heat_df.values, texttemplate="%{text:.0f}%",
            showscale=False))
        st.plotly_chart(style_fig(fig_h, h=350), use_container_width=True)

    sh("Peran & Tipe Mentor")
    m1, m2 = st.columns(2)
    with m1:
        if 'tipe_mentor' in df_f:
            tc = df_f['tipe_mentor'].value_counts().head(6).sort_values(ascending=True)
            fig = px.bar(tc, x=tc.values, y=tc.index, orientation='h', text=tc.values,
                         color_discrete_sequence=[ACCENT])
            fig.update_traces(textposition='outside')
            st.plotly_chart(style_fig(fig, title="Tipe mentor yang diharapkan", h=350), use_container_width=True)
    with m2:
        if 'frekuensi_pembinaan' in df_f:
            fc = df_f['frekuensi_pembinaan'].value_counts()
            fig = px.pie(values=fc.values, names=fc.index, hole=0.5, color_discrete_sequence=COLORWAY)
            st.plotly_chart(style_fig(fig, title="Frekuensi pembinaan ideal per bulan", h=350), use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 6 — FASILITASI KOMPETISI
# ═══════════════════════════════════════════════════════
with t6:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        v = (df_f['pengetahuan_fasilitasi'] == 'Ya').mean() * 100 if 'pengetahuan_fasilitasi' in df_f else None
        kpi("Tahu Fasilitasi", f"{v:.0f}%" if v is not None else "–")
    with k2:
        v = (df_f['pernah_pakai'] == 'Ya').mean() * 100 if 'pernah_pakai' in df_f else None
        kpi("Pernah pakai", f"{v:.0f}%" if v is not None else "–")
    with k3:
        v = df_f['kualitas_layanan'].dropna().astype(float).mean() if 'kualitas_layanan' in df_f else None
        kpi("Kualitas layanan", f"{v:.2f}/4.0" if v is not None else "–")
    with k4:
        v = df_f['promosi_efektif'].dropna().astype(float).mean() if 'promosi_efektif' in df_f else None
        kpi("Keterjangkauan promosi", f"{v:.2f}/4.0" if v is not None else "–")

    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        sh("Sumber Informasi Fasilitasi")
        if 'sumber_info' in df_f:
            sc = df_f['sumber_info'].dropna().str.split(',').explode().str.strip().value_counts().sort_values(ascending=True)
            fig = px.bar(sc, x=sc.values, y=sc.index, orientation='h', text=sc.values,
                         color_discrete_sequence=[ACCENT])
            fig.update_traces(textposition='outside')
            st.plotly_chart(style_fig(fig, h=350), use_container_width=True)
    with f2:
        sh("Media Promosi yang Disukai")
        if 'media_suka' in df_f:
            mc = df_f['media_suka'].dropna().str.split(',').explode().str.strip().value_counts().sort_values(ascending=True)
            fig = px.bar(mc, x=mc.values, y=mc.index, orientation='h', text=mc.values,
                         color_discrete_sequence=['#378ADD'])
            fig.update_traces(textposition='outside')
            st.plotly_chart(style_fig(fig, h=350), use_container_width=True)

    sh("Kendala & Evaluasi")
    e1, e2 = st.columns(2)
    with e1:
        if 'kendala' in df_f:
            kc = df_f['kendala'].dropna().str.split(',').explode().str.strip().value_counts().sort_values(ascending=True).tail(8)
            fig = px.bar(kc, x=kc.values, y=kc.index, orientation='h', text=kc.values,
                         color_discrete_sequence=[ACCENT2])
            fig.update_traces(textposition='outside')
            st.plotly_chart(style_fig(fig, title="Kendala utama", h=350), use_container_width=True)
    with e2:
        if 'evaluasi' in df_f:
            ec = df_f['evaluasi'].dropna().str.split(',').explode().str.strip().value_counts().sort_values(ascending=True).tail(8)
            fig = px.bar(ec, x=ec.values, y=ec.index, orientation='h', text=ec.values,
                         color_discrete_sequence=[ACCENT3])
            fig.update_traces(textposition='outside')
            st.plotly_chart(style_fig(fig, title="Evaluasi yang diusulkan", h=350), use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 7 — INSIGHT TEKS
# ═══════════════════════════════════════════════════════
with t7:
    sh("Lift Antar Minat Belajar (Q1–Q8)")
    q_labels = ['Q1_analisis_data', 'Q2_paper_ilmiah', 'Q3_bisnis_teknologi', 'Q4_coding_AI',
                'Q5_ide_bisnis', 'Q6_pitching', 'Q7_argumen_persuasif', 'Q8_diskusi_sosial']
    short_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
    q_present = [q for q in q_labels if q in df_f.columns]

    if len(q_present) == 8:
        sub_q = df_f[q_labels].dropna()
        if len(sub_q) > 10:
            base_rate = {q: (sub_q[q] == 'Ya').mean() for q in q_labels}
            lift_matrix = np.full((8, 8), np.nan)
            for i, qa in enumerate(q_labels):
                for j, qb in enumerate(q_labels):
                    if i == j:
                        continue
                    a_ya = sub_q[sub_q[qa] == 'Ya']
                    if len(a_ya) > 0 and base_rate[qb] > 0:
                        lift_matrix[i, j] = (a_ya[qb] == 'Ya').mean() / base_rate[qb]
            fig = go.Figure(data=go.Heatmap(
                z=lift_matrix, x=short_labels, y=short_labels, colorscale='RdBu_r',
                zmid=1, text=lift_matrix, texttemplate="%{text:.2f}"))
            st.plotly_chart(style_fig(fig, h=480), use_container_width=True)
            st.caption("Q1: Analisis data · Q2: Paper ilmiah · Q3: Bisnis/tekno · Q4: Coding/AI · "
                       "Q5: Ide bisnis · Q6: Pitching · Q7: Argumen persuasif · Q8: Diskusi sosial")
            ib("Lift >1 berarti asosiasi lebih kuat dari base rate, <1 lebih lemah. Diagonal dikosongkan "
               "karena otomatis bernilai maksimum. Asosiasi terkuat: <b>Q5↔Q6 (ide bisnis ↔ pitching)</b>.")

    sh("Wordcloud Saran Mahasiswa")
    if 'saran' in df_f and WC_OK:
        text_all = " ".join(df_f['saran'].dropna().astype(str).tolist()).lower()
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text_all)
        stopwords = {'yang', 'untuk', 'dengan', 'dari', 'agar', 'akan', 'lebih', 'dapat', 'atau',
                     'saya', 'anda', 'nya', 'para', 'juga', 'bisa', 'tidak', 'ada', 'ini', 'itu', 'dan'}
        words = [w for w in words if w not in stopwords]
        wc1, wc2 = st.columns([1.5, 1])
        with wc1:
            if len(words) > 5:
                wco = WordCloud(width=800, height=380, background_color=bg_panel,
                                 colormap='Greens', max_words=60).generate(" ".join(words))
                fig_wc, ax = plt.subplots(figsize=(8, 3.8), facecolor=bg_panel)
                ax.imshow(wco, interpolation='bilinear')
                ax.axis('off')
                fig_wc.tight_layout(pad=0)
                st.pyplot(fig_wc)
                plt.close()
            else:
                st.info("Teks saran tidak cukup untuk wordcloud.")
        with wc2:
            top_words = Counter(words).most_common(10)
            if top_words:
                wdf = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi']).sort_values('Frekuensi')
                fig_w = px.bar(wdf, x='Frekuensi', y='Kata', orientation='h', color='Frekuensi',
                               color_continuous_scale='Greens')
                fig_w.update_layout(coloraxis_showscale=False)
                st.plotly_chart(style_fig(fig_w, title="Top 10 kata", h=380), use_container_width=True)
    elif not WC_OK:
        st.info("Package `wordcloud` belum terpasang — cek requirements.txt.")

    sh("Jelajahi Saran Mentah")
    if 'saran' in df_f:
        show_df = df_f[['saran']].dropna().rename(columns={'saran': 'Saran mahasiswa'})
        search = st.text_input("Cari kata kunci dalam saran:")
        if search:
            show_df = show_df[show_df['Saran mahasiswa'].str.contains(search, case=False, na=False)]
        st.dataframe(show_df, use_container_width=True, height=400, hide_index=True)
