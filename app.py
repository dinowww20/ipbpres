import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import re
from collections import Counter
from scipy.stats import chi2_contingency, fisher_exact

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
.stApp, p, h1, h2, h3, h4, h5, h6, label, li, .stMarkdown div {{
    font-family: 'Inter', -apple-system, sans-serif !important;
}}
.stApp {{ background: {bg_app} !important; color: {text_main} !important; }}
.main .block-container {{ padding-top: .8rem; padding-bottom: 3rem; max-width: 100%; }}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{ background-color: {bg_panel} !important; border-right: 1px solid {border_col} !important; }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {{ color: {text_main} !important; font-size: 14px; }}
[data-testid="stSidebar"] .stSelectbox>label, [data-testid="stSidebar"] .stMultiSelect>label {{
    color: {text_muted} !important; font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: .6px;
}}
div[data-baseweb="select"] > div {{ background-color: {bg_app} !important; color: {text_main} !important; border-color: {border_col} !important; border-radius: 10px !important; }}
ul[role="listbox"] {{ background-color: {bg_panel} !important; }}
li[role="option"] {{ color: {text_main} !important; background-color: {bg_panel} !important; }}
span[data-baseweb="tag"] {{ background-color: {hover_bg} !important; border: 1px solid {ACCENT} !important; border-radius: 6px !important; }}
span[data-baseweb="tag"] span {{ color: {ACCENT} !important; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{ background-color: {bg_panel} !important; border: 1px solid {border_col}; border-radius: 14px; overflow: hidden; }}

/* ── Charts as elevated cards ── */
[data-testid="stPlotlyChart"] {{
    background: {bg_panel} !important; border: 1px solid {border_col}; border-radius: 16px;
    padding: 14px 10px 6px 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{ background-color: {bg_panel} !important; border-radius: 14px; padding: 6px; gap: 4px; border: 1px solid {border_col} !important; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
.stTabs button[role="tab"] {{ background: transparent !important; border-radius: 10px; padding: 10px 16px; font-weight: 600; font-size: 13px !important; transition: all .15s ease; }}
.stTabs button[role="tab"] p {{ color: {text_muted} !important; }}
.stTabs button[aria-selected="true"] {{ background: linear-gradient(135deg, {ACCENT}, #14876B) !important; box-shadow: 0 3px 8px rgba(15,110,86,0.3); }}
.stTabs button[aria-selected="true"] p {{ color: white !important; font-weight: 700 !important; }}

/* ── KPI cards ── */
.kpi-card {{ background: {bg_panel}; border: 1px solid {border_col}; border-left: 4px solid {ACCENT};
    border-radius: 14px; padding: 18px 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); transition: transform .15s ease; }}
.kpi-label {{ color: {text_muted}; font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; margin-bottom: 8px; }}
.kpi-value {{ color: {text_main}; font-size: 32px; font-weight: 800; line-height: 1.1; letter-spacing: -0.5px; }}
.kpi-sub {{ color: {text_muted}; font-size: 12.5px; margin-top: 5px; }}

/* ── Insight box ── */
.insight-box {{ background: {hover_bg}; border: 1px solid {ACCENT}33; border-left: 4px solid {ACCENT};
    border-radius: 12px; padding: 16px 20px; margin: 12px 0 20px 0; }}
.insight-box p {{ color: {text_main} !important; font-size: 13.8px; margin: 0; line-height: 1.7; }}

/* ── Section titles ── */
.section-title {{ font-size: 18px; font-weight: 700; color: {text_main}; margin: 22px 0 12px 0;
    padding-left: 12px; border-left: 4px solid {ACCENT}; letter-spacing: -0.2px; }}
.element-container {{ margin-bottom: 0 !important; }}
.small-note {{ color: {text_muted}; font-size: 12px; font-style: italic; }}

/* ── Header banner ── */
.dash-header {{ background: linear-gradient(120deg, {ACCENT} 0%, #14876B 60%, #1B9E7E 100%);
    border-radius: 18px; padding: 22px 28px; margin-bottom: 18px; box-shadow: 0 4px 16px rgba(15,110,86,0.25); }}
.dash-header .title {{ font-size: 25px; font-weight: 800; color: white; letter-spacing: -0.3px; }}
.dash-header .subtitle {{ color: rgba(255,255,255,0.85); font-size: 13px; margin-top: 3px; }}
.dash-header .meta {{ color: rgba(255,255,255,0.75); font-size: 12px; text-align: right; line-height: 1.6; }}
.dash-header .meta b {{ color: white; }}
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


def style_fig(fig, title=None, h=380, legend_below=False):
    layout_kwargs = dict(
        template=chart_template, height=h,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color=text_main, size=12.5),
        margin=dict(l=20, r=40, t=45 if title else 15, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5) if legend_below else {},
    )
    if title:
        layout_kwargs['title'] = dict(text=title, font=dict(size=14.5))
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor=border_col, zerolinecolor=border_col, automargin=True)
    fig.update_yaxes(gridcolor=border_col, zerolinecolor=border_col, automargin=True)
    return fig


def shorten(text, n=44):
    """Fallback pemangkas label — dipakai kalau teks tidak ada di dictionary eksplisit."""
    if not isinstance(text, str):
        return text
    text = text.strip()
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def map_short(series, label_dict, fallback_n=44):
    """Terapkan dictionary label singkat, dan pangkas otomatis sisanya yang tidak terdaftar."""
    return series.map(lambda x: label_dict.get(x, shorten(x, fallback_n)) if pd.notna(x) else x)


def explain_stat(p, v, context="", test_name="Chi-square"):
    """Kotak penjelasan awam untuk uji asosiasi & Cramér's V, dengan angka aktual."""
    sig = p < 0.05
    p_txt = (f"karena p-value = <b>{p:.3f}</b> (di bawah 0,05), secara statistik pola ini "
             f"<b>cukup meyakinkan bukan kebetulan acak</b>") if sig else \
            (f"karena p-value = <b>{p:.3f}</b> (di atas 0,05), pola beda-tidaknya "
             f"<b>bisa jadi cuma kebetulan sebaran sampel</b>, bukan pola yang benar-benar nyata")
    if v < 0.1:
        v_txt = "praktis tidak ada keterkaitan"
    elif v < 0.2:
        v_txt = "keterkaitannya sangat lemah"
    elif v < 0.3:
        v_txt = "keterkaitannya lemah-sedang"
    else:
        v_txt = "keterkaitannya cukup kuat"
    return (f"<b>Cara baca ({test_name}):</b> p-value mengukur seberapa besar kemungkinan pola ini muncul "
            f"cuma karena kebetulan acak — semakin kecil p-value (di bawah 0,05), semakin "
            f"meyakinkan bahwa pola itu nyata. Di sini {p_txt}. Cramér's V (0–1) mengukur "
            f"<i>seberapa kuat</i> keterkaitannya — nilai <b>{v:.2f}</b> berarti {v_txt}. "
            f"{context}")


def run_assoc_test(ct):
    """
    Uji asosiasi 2 variabel kategorikal, otomatis pilih metode sesuai kaidah statistik:
    - Tabel 2x2 dengan sel kecil -> Fisher's Exact Test (lebih valid untuk n kecil)
    - Tabel lain -> Chi-square, dengan pengecekan aturan Cochran (maks 20% sel expected count <5,
      tidak ada sel expected count <1). Kalau dilanggar, hasil ditandai caveat eksplisit.
    Mengembalikan dict: {test_name, p, v, valid, warning}
    """
    chi2, p_chi, dof, expected = chi2_contingency(ct)
    nobs = ct.sum().sum()
    v = np.sqrt(chi2 / (nobs * (min(ct.shape) - 1))) if min(ct.shape) > 1 else np.nan

    pct_low = (expected < 5).sum() / expected.size * 100
    any_below_1 = (expected < 1).any()
    cochran_violated = pct_low > 20 or any_below_1

    if ct.shape == (2, 2) and cochran_violated:
        _, p_fisher = fisher_exact(ct.values)
        return {'test_name': "Fisher's Exact Test", 'p': p_fisher, 'v': v,
                'warning': "Tabel 2×2 dengan sel kecil — dipakai Fisher's Exact Test (lebih akurat dari chi-square untuk n kecil), bukan chi-square biasa."}
    elif cochran_violated:
        return {'test_name': "Chi-square", 'p': p_chi, 'v': v,
                'warning': f"Peringatan: {pct_low:.0f}% sel punya expected count di bawah 5 (aturan Cochran dilanggar) — hasil chi-square ini kurang reliabel, baca sebagai indikasi awal saja."}
    else:
        return {'test_name': "Chi-square", 'p': p_chi, 'v': v, 'warning': None}


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
        'Debat & Diplomasi': dict(kompetisi=44, skill=47, target=48, fasilitas=49),
        'ICT–Robotika': dict(kompetisi=51, skill=54, target=55, fasilitas=56),
        'Pengabdian': dict(kompetisi=58, skill=62, target=63, fasilitas=64),
        'Sains & Penalaran': dict(kompetisi=66, skill=69, target=70, fasilitas=71),
        'Bisnis': dict(kompetisi=73, skill=76, target=77, fasilitas=78),
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
    show_debug = st.checkbox("Tampilkan info debug kolom", value=False)
    if show_debug:
        with st.expander(f"Kolom terdeteksi ({len(cmap)-len(missing)}/{len(cmap)})", expanded=True):
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
    'The Active Organizer: Terbukti Berogranisasi dan Berkembang (Aktif dalam organisasi kemahasiswaan (BEM, HIMPRO, UKM) dan Berbagai Kepanitian)': 'The Active Organizer',
    'The Active Organizer: Terbukti Berogranisasi dan Berkembang (Aktif dalam organisasi kemahasiswaan (BEM, HIMPRO, UKM) dan Berbagai Kepanitian': 'The Active Organizer',
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

LABEL_KENDALA = {
    'Waktu proses lama': 'Waktu proses lama',
    'Proses pengajuan atau pencairan fasilitas terlalu panjang dan rumit': 'Proses pengajuan panjang & rumit',
    'Akses fasilitas memerlukan banyak persyaratan atau dokumen administratif': 'Banyak persyaratan/dokumen',
    'Jadwal Operasional Terbatas': 'Jadwal operasional terbatas',
    'Kurang Informasi Mekanisme Pengurusan': 'Kurang info mekanisme',
    'Waktu pembinaan kurang panjang': 'Waktu pembinaan kurang panjang',
    'Minimnya Digitalisasi Pengajuan masih banyak yang harus manual': 'Minim digitalisasi (manual)',
    'Jadwal pendampingan berubah secara tiba-tiba': 'Jadwal berubah tiba-tiba',
    'Petugas administrasi': 'Petugas kurang responsif',
    'kurang responsif': 'Petugas kurang responsif',
    'Pendampingan kurang profesional dan tidak berkesan': 'Pendampingan kurang profesional',
}
LABEL_EVALUASI = {
    'Sosialisasi program ke mahasiswa': 'Sosialisasi program',
    'Kecepatan proses verifikasi dan persetujuan': 'Kecepatan verifikasi',
    'Respons terhadap pertanyaan/ keluhan': 'Respons pertanyaan/keluhan',
    'Kualitas layanan petugas': 'Kualitas layanan petugas',
}
LABEL_SUMBER = {
    'Postingan instagram @ipbprestasi': 'IG @ipbprestasi',
    'Website IPB Prestasi': 'Website IPB Prestasi',
    'Teman': 'Teman',
    'Informasi yang disebarkan melalui WhatsApp di grup angkatan': 'WhatsApp grup angkatan',
    'Postingan instagram BEM KM/BEM Fakultas/Himpunan': 'IG BEM/Himpunan',
    'Website BEM KM/BEM Fakultas/Himpunan': 'Website BEM/Himpunan',
    'Postingan instagram UKM': 'IG UKM',
    'Website Fakultas/Program Studi': 'Website Fakultas/Prodi',
    'Postingan tiktok': 'TikTok',
    'Dosen': 'Dosen',
    'Media cetak (poster) yang ditempel di setiap fakultas': 'Poster cetak',
}
LABEL_MEDIA = {
    'Feed instagram': 'Feed Instagram', 'Instastory': 'Instastory',
    'Channel di WA atau IG': 'Channel WA/IG', 'Melalui jaringan komunikasi di grup angkatan': 'Grup angkatan',
    'Sosialisasi secara offline': 'Sosialisasi offline', 'Video pencerdasan': 'Video edukasi',
    'Video tren': 'Video tren', 'TikTok': 'TikTok', 'Poster': 'Poster', 'Website': 'Website',
}
LABEL_MENTOR = {
    'Kakak tingkat/Alumni yang pernah menjuarai di bidang lomba yang sama': 'Kating/alumni juara sebidang',
    'Dosen yang ahli di bidang terkait': 'Dosen ahli terkait',
    'Mentor profesional/eksternal dari luar kampus di bidang terkait': 'Mentor profesional eksternal',
    'Seangkatan (teman sebaya) yang sudah lebih berpengalaman': 'Teman seangkatan berpengalaman',
}
LABEL_KRITERIA_MENTOR = {
    'Rutin memberi arahan dan masukan': 'Rutin memberi arahan',
    'Fleksibel dan terbuka diskusi': 'Fleksibel & terbuka diskusi',
    'Memberi contoh ril terkait materi lomba': 'Kasih contoh riil materi lomba',
    'Tegas, disiplin, tapi suportif': 'Tegas, disiplin, suportif',
    'Memberi motivasi dan semangat': 'Memberi motivasi & semangat',
    'Santai dan dapat menjadi teman mengobrol di luar konteks kegiatan/lomba': 'Santai, bisa jadi teman ngobrol',
    'Sering berbagi pengalaman pribadi': 'Sering berbagi pengalaman',
}
LABEL_SKILL_BIDANG = {
    'Kepercayaan diri saat berbicara di depan umum': 'Percaya diri bicara depan umum',
    'Kemampuan menyusun argumen secara sistematis': 'Menyusun argumen sistematis',
    'Kemampuan berpikir kritis, logis dan tepat': 'Berpikir kritis & logis',
    'Kemampuan pemrograman (coding)': 'Kemampuan pemrograman (coding)',
    'Kepercayaan diri untuk mulai mencoba membuat proyek': 'Percaya diri mulai proyek',
    'Kemampuan kerja sama dalam tim teknis': 'Kerja sama tim teknis',
    'Pemahaman dasar elektronika dan mekanik': 'Dasar elektronika & mekanik',
    'Pemahaman terhadap isu sosial, budaya, dan kondisi lokal': 'Pemahaman isu sosial & lokal',
    'Kemampuan pemecahan masalah terhadap isu sosial sekitar': 'Pemecahan masalah isu sosial',
    'Kemampuan komunikasi langsung dengan masyarakat': 'Komunikasi langsung masyarakat',
    'Konsistensi dalam kegiatan jangka panjang': 'Konsistensi jangka panjang',
    'Kesiapan fisik dan mental untuk turun ke lapangan': 'Kesiapan fisik & mental',
    'Konsistensi dalam mengembangkan ide & membaca sumber terpercaya': 'Konsistensi ide & baca sumber',
    'Kemampuan berpikir dalam mengerjakan soal-soal sulit': 'Berpikir soal-soal sulit',
    'Kemampuan penalaran logis dan analisis argumen': 'Penalaran logis & analisis',
    'Kemampuan memecahkan masalah secara sistematis': 'Pemecahan masalah sistematis',
    'Kemampuan mengambil keputusan berbasis data dan bukti': 'Keputusan berbasis data',
    'Keterbukaan terhadap masukan dan kritik ilmiah': 'Terbuka kritik ilmiah',
    'Manajemen keuangan dasar dan permodalan': 'Manajemen keuangan & modal',
    'Kemampuan merancang produk/jasa yang relevan': 'Merancang produk/jasa',
    'Keberanian memulai usaha dari nol': 'Berani mulai usaha dari nol',
    'Konsistensi dan komitmen menjalankan bisnis': 'Konsisten menjalankan bisnis',
    'Strategi pemasaran dan branding di media sosial': 'Pemasaran & branding medsos',
}
LABEL_FASILITAS_BIDANG = {
    'Pelatihan debat/diplomasi yang aktif dan konsisten': 'Pelatihan debat aktif & konsisten',
    'Kolaborasi antar fakultas untuk kegiatan debat lintas jurusan': 'Kolaborasi debat lintas jurusan',
    'Pendampingan dari dosen atau alumni yang lebih kompeten': 'Pendampingan dosen/alumni kompeten',
    'Workshop teknis pemula secara berkala': 'Workshop teknis pemula berkala',
    'Komunitas robotika yang terbuka bagi pemula': 'Komunitas robotika terbuka',
    'Wadah showcase untuk karya/proyek robotik': 'Wadah showcase karya robotik',
    'Akses program pengabdian dengan pendampingan yang baik': 'Akses program & pendampingan baik',
    'Pengakuan formal (SKPI, sertifikat, konversi kredit)': 'Pengakuan formal (SKPI/sertifikat)',
    'Pembekalan teknis dan non-teknis sebelum turun ke lapangan': 'Pembekalan sebelum ke lapangan',
    'Kemitraan dengan desa/kelompok masyarakat yang konsisten': 'Kemitraan desa konsisten',
    'Pendanaan atau hibah untuk inisiasi program mandiri': 'Pendanaan program mandiri',
    'Bantuan logistik dan transportasi ke lokasi pengabdian': 'Bantuan logistik & transportasi',
    'Program pelatihan penalaran logis & critical thinking': 'Pelatihan penalaran & critical thinking',
    'Kelas atau kegiatan untuk latihan pemecahan masalah berbasis kasus nyata': 'Latihan kasus nyata',
    'Pendanaan untuk riset atau lomba akademik': 'Pendanaan riset/lomba akademik',
    'Workshop ide riset atau proposal dari dosen/alumni': 'Workshop ide riset/proposal',
    'Penyediaan ruang latihan atau simulasi debat': 'Ruang latihan/simulasi debat',
    'Akses ke pelatihan wirausaha dengan profesional': 'Pelatihan wirausaha profesional',
    'Inkubator bisnis dengan pendampingan intensif': 'Inkubator bisnis intensif',
    'Pendanaan untuk ide bisnis yang dimiliki': 'Pendanaan ide bisnis',
    'Pendampingan dari alumni yang sudah sukses berbisnis': 'Pendampingan alumni sukses',
    'Workshop digital marketing dan legalitas usaha': 'Workshop digital marketing & legalitas',
}


LABEL_TARGET_BIDANG = {
    'Mengembangkan skill public speaking dan berpikir cepat': 'Skill public speaking & cepat',
    'Mahasiswa Berprestasi': 'Jadi mahasiswa berprestasi',
    'Menang lomba debat di tingkat regional/nasional': 'Menang lomba debat regional/nasional',
    'Menjadi perwakilan kampus dalam forum diplomasi (seperti MUN)': 'Perwakilan kampus forum diplomasi',
    'Membangun portofolio prestasi untuk beasiswa atau karier': 'Portofolio untuk beasiswa/karier',
    'Mengembangkan portofolio proyek untuk karier di bidang teknologi': 'Portofolio proyek karier teknologi',
    'Masih eksplorasi, belum punya target pasti': 'Masih eksplorasi',
    'Mengikuti dan menang dalam lomba robotika': 'Menang lomba robotika',
    'Bergabung dalam tim robotika resmi kampus': 'Gabung tim robotika kampus',
    'Menyusun dan menjalankan program sosial yang berdampak nyata bagi  masyarakat': 'Program sosial berdampak nyata',
    'Mengikuti kompetisi di bidang pengabdian masyarakat (PPKO, PKM PM, dll)': 'Kompetisi pengabdian (PPKO/PKM)',
    'Belum punya target, masih ingin belajar dan ikut serta': 'Belum punya target, ingin belajar',
    'Terlibat aktif dalam program desa binaan': 'Terlibat program desa binaan',
    'Mendapatkan dan berkolaborasi dengan mitra eksternal (LSM, CSR, dll.)': 'Kolaborasi mitra eksternal',
    'Menang dalam kompetisi (LKTI, esai, riset, Olimpiade, debat ilmiah)': 'Menang kompetisi ilmiah (LKTI/riset)',
    'Mempublikasikan karya atau gagasan di media ilmiah/populer kredibel': 'Publikasi karya ilmiah',
    'Belum punya target pasti, masih eksplorasi': 'Belum punya target, eksplorasi',
    'Meningkatkan kemampuan berpikir kritis terukur (misal asesmen/tes)': 'Berpikir kritis terukur',
    'Mengikuti konferensi atau forum ilmiah mahasiswa': 'Konferensi/forum ilmiah',
    'Menang kompetisi ide bisnis atau proposal usaha (Bussiness Plan Competition, Bussines Case Competition, Bussiness Model Canvas, dll)': 'Menang kompetisi ide bisnis',
    'Memulai bisnis sendiri meskipun kecil': 'Memulai bisnis sendiri',
    'Bergabung di inkubator bisnis mahasiswa': 'Gabung inkubator bisnis',
    'Masih tahap belajar, ingin tahu lebih dulu': 'Masih tahap belajar',
    'Mengembangkan usaha yang sudah berjalan': 'Kembangkan usaha berjalan',
}


def normalize_frekuensi(x):
    if pd.isna(x):
        return x
    x = str(x).strip().lower().replace(' ', '')
    m = re.match(r'(\d+)', x)
    if not m:
        return None
    n = int(m.group(1))
    if n > 8:
        return None  # outlier tidak masuk akal (ex: "10 kali")
    return f"{n}x/bulan"

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
st.markdown(f"""<div class='dash-header' style='display:flex; justify-content:space-between; align-items:center;'>
<div><div class='title'>Dashboard Analisis Survey Prestasi</div>
<div class='subtitle'>IPB University — Fasilitasi Kompetitif Mahasiswa · Purposive sampling, 4 Feb–21 Mar 2026</div></div>
<div class='meta'>n = <b>{N}</b> responden{" (terfilter)" if N != len(df) else f" dari {len(df)}"}<br>Skala kepuasan 1–4</div>
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

    ov1, ov2 = st.columns([1.3, 1])
    with ov1:
        sh("Profil Mahasiswa Selama Perkuliahan")
        if 'profil_mhs' in df_f:
            pc = map_short(df_f['profil_mhs'], LABEL_PROFIL).value_counts().sort_values(ascending=True)
            fig = px.bar(pc, x=pc.values, y=pc.index, orientation='h', text=pc.values,
                         color=pc.values, color_continuous_scale='Teal')
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, h=340), use_container_width=True)
            top = pc.idxmax()
            ib(f"Profil dominan adalah <b>{top}</b> ({pc.max()} orang, {pc.max()/pc.sum()*100:.0f}%) — "
               f"mayoritas mahasiswa mengejar keseimbangan antara akademik dan aktivitas non-akademik, bukan fokus tunggal.")
    with ov2:
        sh("Snapshot Karakteristik")
        radar_vals = []
        radar_labels = []
        for label, val in [("Adaptasi cepat", pct_cepat), ("Berminat kompetisi", pct_minat),
                            ("Prefer kerja tim", pct_tim), ("Tahu Fasilitasi", pct_tahu)]:
            if val is not None:
                radar_labels.append(label)
                radar_vals.append(val)
        if radar_vals:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=radar_vals + [radar_vals[0]],
                                             theta=radar_labels + [radar_labels[0]],
                                             fill='toself', line_color=ACCENT, fillcolor=ACCENT,
                                             opacity=0.75))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor=border_col)),
                                 showlegend=False)
            st.plotly_chart(style_fig(fig_r, h=340), use_container_width=True)
            st.caption("Setiap sumbu = persentase dari total responden (0–100%).")

    sh("Sebaran Bidang Minat Utama")
    if 'bidang_minat_utama' in df_f:
        bc = df_f['bidang_minat_utama'].value_counts()
        bc = bc[bc.index.isin(BIDANG_MAP_RAW.values())]
        fig = px.pie(values=bc.values, names=bc.index, hole=0.5, color_discrete_sequence=COLORWAY)
        fig.update_traces(textinfo='label+percent')
        st.plotly_chart(style_fig(fig, h=380), use_container_width=True)

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
            jc = df_f['jalur_masuk'].value_counts()
            fig = px.treemap(names=jc.index, parents=[""] * len(jc), values=jc.values,
                              color=jc.values, color_continuous_scale='Blues')
            fig.update_traces(textinfo='label+value+percent root', textfont_size=13)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, h=350), use_container_width=True)
    with d4:
        sh("Distribusi Angkatan")
        if 'angkatan' in df_f:
            ac = df_f['angkatan'].value_counts().sort_index(ascending=False)
            fig = px.bar(x=ac.index.astype(str), y=ac.values, text=ac.values,
                         color=ac.values, color_continuous_scale='Purp')
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Angkatan", yaxis_title="Jumlah mahasiswa")
            st.plotly_chart(style_fig(fig, h=350), use_container_width=True)

    sh("Komunitas yang Pernah/Sedang Diikuti")
    if 'komunitas' in df_f:
        ks = df_f['komunitas'].dropna().str.split(',').explode().str.strip()
        ks = map_short(ks, LABEL_KOMUNITAS)
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
            st.plotly_chart(style_fig(fig, h=360, legend_below=True), use_container_width=True)
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
            st.plotly_chart(style_fig(fig2, h=360, legend_below=True), use_container_width=True)
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
        col = f'{b}__kompetisi'
        if col in df_f.columns:
            s = df_f[col].dropna()
            n = len(s)
            ya = (s == 'Ya').sum()
            rows.append({'Bidang': b, 'n': n, 'Ya': ya, 'pct': ya / n * 100 if n > 0 else 0})
    if rows:
        bidang_df = pd.DataFrame(rows).sort_values('pct')
        fig = px.scatter(bidang_df, x='pct', y='Bidang', size='n', color='pct',
                          color_continuous_scale='Teal', size_max=55,
                          text=[f"{p:.0f}%" for p in bidang_df['pct']])
        fig.update_traces(textposition='middle right', textfont_size=12)
        fig.update_layout(xaxis_range=[0, 100], xaxis_title="% pernah ikut kompetisi", coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, h=330), use_container_width=True)
        ib("Ukuran lingkaran = jumlah responden (n) per bidang — bidang dengan lingkaran kecil "
           "(Debat, ICT) perlu dibaca hati-hati karena basis datanya tipis.")
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
                res = run_assoc_test(ct)
                ctp = ct.div(ct.sum(axis=1), axis=0) * 100
                ctp = ctp.loc[ctp.get('Ya', pd.Series(dtype=float)).sort_values(ascending=True).index] if 'Ya' in ctp else ctp
                fig = go.Figure()
                for cat, color in zip(['Tidak', 'Ya'], [ACCENT2, ACCENT]):
                    if cat in ctp.columns:
                        fig.add_trace(go.Bar(y=ctp.index, x=ctp[cat], orientation='h', name=cat,
                                              marker_color=color, text=[f"{v:.0f}%" for v in ctp[cat]], textposition='inside'))
                fig.update_layout(barmode='stack')
                st.plotly_chart(style_fig(fig, title="Jalur Masuk × Kecepatan Adaptasi", h=300, legend_below=True), use_container_width=True)
                if res['warning']:
                    st.markdown(f"<span class='small-note'>⚠ {res['warning']}</span>", unsafe_allow_html=True)
                ib(explain_stat(res['p'], res['v'], "Kesimpulan: jalur masuk mahasiswa <b>tidak berkaitan</b> dengan cepat-tidaknya beradaptasi.", res['test_name']))

    with cb:
        if all(k in df_f for k in ['bidang_minat_utama', 'kerja_sendiri']):
            valid = df_f[df_f['bidang_minat_utama'].isin(BIDANG_MAP_RAW.values())]
            ct2 = pd.crosstab(valid['bidang_minat_utama'], valid['kerja_sendiri'])
            if ct2.shape[0] > 1 and ct2.shape[1] > 1:
                res2 = run_assoc_test(ct2)
                ct2p = ct2.div(ct2.sum(axis=1), axis=0) * 100
                fig2 = go.Figure()
                for cat, color, lbl in zip(['Tidak', 'Ya'], ['#7F77DD', '#378ADD'], ['Lebih suka tim', 'Lebih suka sendiri']):
                    if cat in ct2p.columns:
                        fig2.add_trace(go.Bar(y=ct2p.index, x=ct2p[cat], orientation='h', name=lbl,
                                               marker_color=color, text=[f"{v:.0f}%" for v in ct2p[cat]], textposition='inside'))
                fig2.update_layout(barmode='stack')
                st.plotly_chart(style_fig(fig2, title="Bidang Minat × Gaya Kerja", h=300, legend_below=True), use_container_width=True)
                if res2['warning']:
                    st.markdown(f"<span class='small-note'>⚠ {res2['warning']}</span>", unsafe_allow_html=True)
                ib(explain_stat(res2['p'], res2['v'], "Kesimpulan: bidang minat <b>tidak berkaitan kuat</b> dengan preferensi kerja sendiri/tim.", res2['test_name']))

    sh("Fasilitas & Skill Paling Diprioritaskan per Bidang")
    fas_rows = []
    for b in BIDANG_LIST:
        col_f = f'{b}__fasilitas'
        col_s = f'{b}__skill'
        col_t = f'{b}__target'
        if col_f in df_f.columns:
            vc = df_f[col_f].value_counts()
            vc_s = df_f[col_s].value_counts() if col_s in df_f.columns else pd.Series(dtype=int)
            vc_t = df_f[col_t].value_counts() if col_t in df_f.columns else pd.Series(dtype=int)
            if len(vc) > 0:
                top_f = LABEL_FASILITAS_BIDANG.get(vc.index[0], shorten(vc.index[0], 42))
                top_s = LABEL_SKILL_BIDANG.get(vc_s.index[0], shorten(vc_s.index[0], 42)) if len(vc_s) > 0 else "–"
                top_t = LABEL_TARGET_BIDANG.get(vc_t.index[0], shorten(vc_t.index[0], 42)) if len(vc_t) > 0 else "–"
                fas_rows.append({'Bidang': b, 'n': vc.sum(),
                                  'Target utama': top_t,
                                  'Skill paling dibutuhkan': top_s,
                                  'Fasilitas paling diharapkan': top_f, '%': f"{vc.iloc[0]/vc.sum()*100:.0f}%"})
    if fas_rows:
        st.dataframe(pd.DataFrame(fas_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
# TAB 5 — POLA PEMBINAAN
# ═══════════════════════════════════════════════════════
with t5:
    sh("Preferensi Format Pembinaan")
    label_ukuran_kelompok = {
        'Kelompok Kecil (Hanya 1 tim/ maksimal 5 orang)': 'Kelompok kecil',
        '1 on 1 atau Personal (hanya diri sendiri)': 'Personal (1-on-1)',
        'Kelompok Besar (Pembinaan kelas seperti enrichment dsb.)': 'Kelompok besar',
    }
    label_aktivitas = {
        'Pendampingan dan Pembimbingan Personal (Mentoring/coaching dengan dosen/alumni/tokoh inspiratif)': 'Coaching personal',
        'Penguatan Relasi dan Keakraban(Sikrab/makrab, diskusi santai, dan studi kasus)': 'Relasi & keakraban',
        'Pelatihan Keterampilan dan Simulasi Lapangan (Simulasi lomba dan Praktik )': 'Simulasi & praktik',
        'Penguatan Pengetahuan dan Pemahaman Teoritis (Training dan workshop)': 'Teori (workshop)',
    }
    label_struktur = {
        'Pembinaan bebas (fleksibel) dan tidak terafiliasi dengan organisasi formal kampus, bisa diikuti tanpa komitmen keanggotaan atau struktur formal': 'Fleksibel',
        'Pembinaan terstruktur (terikat) dan diselenggarakan oleh pihak yang berafiliasi dengan organisasi formal kampus , dan biasanya memerlukan komitmen keikutsertaan': 'Terstruktur',
    }
    label_metode_pelaksanaan = {'Hybrid': 'Hybrid', 'Luring': 'Luring', 'Daring': 'Daring'}

    def pct_series(col, lmap):
        if col not in df_f.columns:
            return pd.Series(dtype=float)
        s = df_f[col].dropna().map(lambda x: lmap.get(x, x)).value_counts()
        return (s / s.sum() * 100).round(0) if s.sum() > 0 else s

    # Setiap baris dipasangkan dengan kolom & dictionary yang SESUAI isinya (tervalidasi ke data asli)
    rows_pref = {
        'Metode pelaksanaan': pct_series('metode_pembinaan', label_metode_pelaksanaan),
        'Ukuran kelompok': pct_series('ukuran_kelompok', label_ukuran_kelompok),
        'Aktivitas ideal': pct_series('aktivitas_pembinaan', label_aktivitas),
        'Struktur program': pct_series('struktur_program', label_struktur),
    }
    fig = go.Figure()
    for row_name, series in rows_pref.items():
        left = 0
        for j, (cat, val) in enumerate(series.items()):
            fig.add_trace(go.Bar(y=[row_name], x=[val], orientation='h', name=cat,
                                  marker_color=COLORWAY[j % len(COLORWAY)],
                                  text=f"{cat} {val:.0f}%", textposition='inside',
                                  showlegend=False, base=left))
            left += val
    fig.update_layout(barmode='overlay', xaxis_range=[0, 100])
    st.plotly_chart(style_fig(fig, h=360), use_container_width=True)
    ib("Mayoritas menginginkan pembinaan <b>hybrid</b>, dalam <b>kelompok kecil</b>, berupa "
       "<b>coaching personal</b> dengan dosen/alumni, dan format <b>fleksibel</b> tanpa keterikatan organisasi formal.")

    sh("Pola Preferensi Pembinaan per Bidang Minat")
    if 'bidang_minat_utama' in df_f and all(c in df_f for c in ['ukuran_kelompok', 'aktivitas_pembinaan', 'struktur_program']):
        sub = df_f[df_f['bidang_minat_utama'].isin(BIDANG_MAP_RAW.values())].copy()
        sub['ukuran_short'] = sub['ukuran_kelompok'].map(label_ukuran_kelompok)
        sub['aktivitas_short'] = sub['aktivitas_pembinaan'].map(label_aktivitas)
        sub['struktur_short'] = sub['struktur_program'].map(label_struktur)

        heat_rows = []
        for b_raw in sub['bidang_minat_utama'].unique():
            g = sub[sub['bidang_minat_utama'] == b_raw]
            n = len(g)
            heat_rows.append({
                'Bidang': b_raw, 'n': n,
                '% Kelompok kecil': (g['ukuran_short'] == 'Kelompok kecil').sum() / g['ukuran_short'].notna().sum() * 100 if g['ukuran_short'].notna().sum() else np.nan,
                '% Coaching personal': (g['aktivitas_short'] == 'Coaching personal').sum() / g['aktivitas_short'].notna().sum() * 100 if g['aktivitas_short'].notna().sum() else np.nan,
                '% Fleksibel': (g['struktur_short'] == 'Fleksibel').sum() / g['struktur_short'].notna().sum() * 100 if g['struktur_short'].notna().sum() else np.nan,
            })
        heat_df = pd.DataFrame(heat_rows).set_index('Bidang')
        n_col = heat_df.pop('n')
        heat_df = heat_df.round(0)

        hcol1, hcol2 = st.columns([1, 1])
        with hcol1:
            fig_h = go.Figure(data=go.Heatmap(
                z=heat_df.values, x=heat_df.columns, y=[f"{i} (n={n_col[i]})" for i in heat_df.index],
                colorscale='Blues', text=heat_df.values, texttemplate="%{text:.0f}%",
                showscale=False))
            st.plotly_chart(style_fig(fig_h, h=340), use_container_width=True)
        with hcol2:
            fig_radar = go.Figure()
            theta = list(heat_df.columns) + [heat_df.columns[0]]
            for i, (bidang, row) in enumerate(heat_df.iterrows()):
                vals = list(row.values) + [row.values[0]]
                fig_radar.add_trace(go.Scatterpolar(r=vals, theta=theta, name=bidang,
                                                      line_color=COLORWAY[i % len(COLORWAY)], opacity=0.85))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor=border_col)),
                                     legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, font=dict(size=10)))
            st.plotly_chart(style_fig(fig_radar, h=340), use_container_width=True)
        ib("Heatmap dan radar menunjukkan data yang sama, dua sudut pandang berbeda: heatmap untuk baca "
           "angka presisi per sel, radar untuk lihat sekilas bidang mana yang \"bentuk preferensinya\" mirip satu sama lain.")

    sh("Peran & Tipe Mentor")
    m1, m2 = st.columns(2)
    with m1:
        if 'tipe_mentor' in df_f:
            tc = df_f['tipe_mentor'].dropna().str.split(',').explode().str.strip()
            tc = map_short(tc, LABEL_MENTOR)
            tc = tc.value_counts()
            fig = px.treemap(names=tc.index, parents=[""] * len(tc), values=tc.values,
                              color=tc.values, color_continuous_scale='Purp')
            fig.update_traces(textinfo='label+value', textfont_size=12)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, title="Tipe mentor yang diharapkan (multi-select)", h=350), use_container_width=True)
    with m2:
        if 'frekuensi_pembinaan' in df_f:
            fc = df_f['frekuensi_pembinaan'].map(normalize_frekuensi).dropna().value_counts()
            fig = px.pie(values=fc.values, names=fc.index, hole=0.5, color_discrete_sequence=COLORWAY)
            fig.update_traces(textinfo='label+percent')
            st.plotly_chart(style_fig(fig, title="Frekuensi pembinaan ideal per bulan", h=350), use_container_width=True)

    sh("Kriteria Mentor \"Ideal\" Menurut Mahasiswa")
    if 'kriteria_mentor' in df_f:
        # opsi "Tegas, disiplin, tapi suportif" mengandung koma internal — lindungi dulu sebelum split
        placeholder = "Tegas|disiplin|tapi suportif"
        krc_raw = df_f['kriteria_mentor'].dropna().str.replace(
            'Tegas, disiplin, tapi suportif', placeholder, regex=False)
        krc = krc_raw.str.split(',').explode().str.strip()
        krc = krc.str.replace(placeholder, 'Tegas, disiplin, tapi suportif', regex=False)
        krc = map_short(krc, LABEL_KRITERIA_MENTOR)
        krc = krc.value_counts().sort_values(ascending=True)
        fig = px.bar(krc, x=krc.values, y=krc.index, orientation='h', text=krc.values,
                     color=krc.values, color_continuous_scale='Purp')
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, h=380), use_container_width=True)
        ib("Multi-select (maksimal 3 pilihan per responden) — <b>rutin memberi arahan</b> dan "
           "<b>fleksibel/terbuka diskusi</b> jadi dua kriteria paling banyak diminta, mengalahkan "
           "faktor \"galak vs santai\" yang sering diasumsikan penting.")

# ═══════════════════════════════════════════════════════
# TAB 6 — FASILITASI KOMPETISI
# ═══════════════════════════════════════════════════════
with t6:
    k1, k2 = st.columns(2)
    with k1:
        v = (df_f['pengetahuan_fasilitasi'] == 'Ya').mean() * 100 if 'pengetahuan_fasilitasi' in df_f else None
        kpi("Tahu Fasilitasi", f"{v:.0f}%" if v is not None else "–", "minimal 1 program")
    with k2:
        v = (df_f['pernah_pakai'] == 'Ya').mean() * 100 if 'pernah_pakai' in df_f else None
        kpi("Pernah pakai fasilitasi", f"{v:.0f}%" if v is not None else "–", "dari total responden")

    g1, g2, g3, g4 = st.columns(4)
    gauge_data = [
        ('kualitas_layanan', 'Kualitas layanan', g1),
        ('promosi_efektif', 'Keterjangkauan promosi', g2),
        ('kemudahan_alur', 'Kemudahan alur', g3),
        ('kecepatan_verif', 'Kecepatan verifikasi', g4),
    ]
    for col_key, label, container in gauge_data:
        with container:
            if col_key in df_f:
                val = df_f[col_key].dropna().astype(float).mean()
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=val, number={'suffix': "/4", 'font': {'size': 24}},
                    gauge={'axis': {'range': [0, 4], 'tickwidth': 1},
                           'bar': {'color': ACCENT if val >= 2.5 else ACCENT2},
                           'steps': [{'range': [0, 2], 'color': hover_bg}, {'range': [2, 4], 'color': bg_panel}],
                           'threshold': {'line': {'color': text_muted, 'width': 2}, 'value': 2.5}}))
                st.plotly_chart(style_fig(fig, title=label, h=220), use_container_width=True)
    ib("Skala 1–4. Nilai di bawah 2,5 (garis abu-abu) menandakan area yang perlu diperbaiki lebih dulu.")

    f1, f2 = st.columns(2)
    with f1:
        sh("Sumber Informasi Fasilitasi")
        if 'sumber_info' in df_f:
            sc = df_f['sumber_info'].dropna().str.split(',').explode().str.strip()
            sc = map_short(sc, LABEL_SUMBER).value_counts().sort_values(ascending=True)
            fig = px.bar(sc, x=sc.values, y=sc.index, orientation='h', text=sc.values,
                         color=sc.values, color_continuous_scale='Teal')
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, h=350), use_container_width=True)
    with f2:
        sh("Media Promosi yang Disukai")
        if 'media_suka' in df_f:
            mc = df_f['media_suka'].dropna().str.split(',').explode().str.strip()
            mc = map_short(mc, LABEL_MEDIA).value_counts().sort_values(ascending=True)
            fig = px.bar(mc, x=mc.values, y=mc.index, orientation='h', text=mc.values,
                         color=mc.values, color_continuous_scale='Blues')
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, h=350), use_container_width=True)
    ib("Instagram (feed & story) dan komunikasi grup angkatan mendominasi baik sebagai sumber info yang "
       "sudah efektif maupun sebagai media yang paling disukai — konsisten, jadi prioritas jelas untuk promosi ke depan.")

    sh("Kendala & Evaluasi yang Diusulkan")
    e1, e2 = st.columns(2)
    with e1:
        if 'kendala' in df_f:
            kc = df_f['kendala'].dropna().str.split(',').explode().str.strip()
            kc = kc[kc.str.len() > 2]
            kc = map_short(kc, LABEL_KENDALA).value_counts().sort_values(ascending=True).tail(8)
            fig = px.bar(kc, x=kc.values, y=kc.index, orientation='h', text=kc.values,
                         color=kc.values, color_continuous_scale='Oranges')
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, title="Kendala utama", h=360), use_container_width=True)
    with e2:
        if 'evaluasi' in df_f:
            ec = df_f['evaluasi'].dropna().str.split(',').explode().str.strip()
            ec = ec[ec.isin(LABEL_EVALUASI.keys())]
            ec = map_short(ec, LABEL_EVALUASI).value_counts().sort_values(ascending=True)
            fig = px.bar(ec, x=ec.values, y=ec.index, orientation='h', text=ec.values,
                         color=ec.values, color_continuous_scale='Purp')
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, title="Evaluasi yang diusulkan", h=360), use_container_width=True)

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
            text_matrix = np.full((8, 8), "", dtype=object)
            for i, qa in enumerate(q_labels):
                for j, qb in enumerate(q_labels):
                    if i == j:
                        continue
                    a_ya = sub_q[sub_q[qa] == 'Ya']
                    if len(a_ya) > 0 and base_rate[qb] > 0:
                        lift_val = (a_ya[qb] == 'Ya').mean() / base_rate[qb]
                        lift_matrix[i, j] = lift_val
                        pair_ct = pd.crosstab(sub_q[qa], sub_q[qb])
                        sig_mark = ""
                        if pair_ct.shape == (2, 2):
                            _, p_pair = fisher_exact(pair_ct.values)
                            sig_mark = "*" if p_pair < 0.05 else ""
                        text_matrix[i, j] = f"{lift_val:.2f}{sig_mark}"
            fig = go.Figure(data=go.Heatmap(
                z=lift_matrix, x=short_labels, y=short_labels, colorscale='RdBu_r',
                zmid=1, text=text_matrix, texttemplate="%{text}"))
            st.plotly_chart(style_fig(fig, h=460), use_container_width=True)
            st.caption("Q1: Analisis data · Q2: Paper ilmiah · Q3: Bisnis/tekno · Q4: Coding/AI · "
                       "Q5: Ide bisnis · Q6: Pitching · Q7: Argumen persuasif · Q8: Diskusi sosial · "
                       "* = signifikan secara statistik (Fisher's Exact Test, p<0,05)")

            flat = [(short_labels[i], short_labels[j], lift_matrix[i, j])
                    for i in range(8) for j in range(8) if i != j and not np.isnan(lift_matrix[i, j])]
            top_pair = max(flat, key=lambda x: x[2])
            low_pair = min(flat, key=lambda x: x[2])
            pct_up = (top_pair[2] - 1) * 100
            pct_down = (1 - low_pair[2]) * 100
            ib(f"<b>Cara baca lift:</b> angka 1,00 berarti dua minat itu independen (tidak saling terkait). "
               f"Di atas 1 berarti saling memperkuat, di bawah 1 berarti saling melemahkan. Tanda <b>*</b> "
               f"berarti keterkaitannya sudah diuji signifikan secara statistik (bukan cuma kebetulan sampel) — "
               f"<b>abaikan angka tanpa tanda *</b>, itu belum cukup bukti untuk disimpulkan sebagai pola nyata. "
               f"Contoh: lift <b>{top_pair[0]}→{top_pair[1]} = {top_pair[2]:.2f}</b> artinya mahasiswa yang tertarik "
               f"{top_pair[0]} punya peluang <b>{pct_up:.0f}% lebih tinggi</b> dari rata-rata untuk juga "
               f"tertarik {top_pair[1]}. Sebaliknya, lift <b>{low_pair[0]}→{low_pair[1]} = {low_pair[2]:.2f}</b> "
               f"artinya peluangnya <b>{pct_down:.0f}% lebih rendah</b> dari rata-rata.")

    sh("Program Fasilitasi yang Paling Dikenal")
    if 'program_diketahui' in df_f:
        pd_short = {
            'Fasilitasi Pendanaan Kompetitif Mandiri Nasional & International': 'Pendanaan Kompetitif',
            'Fasilitasi Administratif Kompetitif Mandiri Nasional & International': 'Administratif Kompetitif',
            'Fasilitasi Pembinaan Kompetitif Mandiri Nasional & International': 'Pembinaan Kompetitif',
            'Fasilitasi Insentif Prestasi Kompetitif Mandiri': 'Insentif Prestasi',
        }
        pdc = df_f['program_diketahui'].dropna().str.split(',').explode().str.strip()
        pdc = map_short(pdc, pd_short).value_counts()
        fig = px.treemap(names=pdc.index, parents=[""] * len(pdc), values=pdc.values,
                          color=pdc.values, color_continuous_scale='Greens')
        fig.update_traces(textinfo='label+value+percent root', textfont_size=13)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, h=320), use_container_width=True)

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
