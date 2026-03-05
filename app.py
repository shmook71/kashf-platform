import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# =========================
# إعداد الصفحة
# =========================
st.set_page_config(page_title="منصة كشف", layout="wide")

# =========================
# ثيم (Deep Teal + Mint) + RTL + تصميم رسمي + خط عربي
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;600;700&display=swap');

:root{
  --bgTop: #DDEFE1;
  --bgMid: #2A6B5E;
  --bgBot: #0B2E2A;

  --card: rgba(255,255,255,0.10);
  --card2: rgba(255,255,255,0.14);
  --border: rgba(255,255,255,0.18);

  --text: #EAF7EE;
  --muted: rgba(234,247,238,0.78);

  --radius: 18px;
  --shadow: 0 12px 32px rgba(0,0,0,0.26);
}

html, body, [class*="css"] {
  direction: rtl;
  text-align: right;
  font-family: "IBM Plex Sans Arabic", system-ui, -apple-system, Segoe UI, sans-serif !important;
}

.stApp{
  background: linear-gradient(180deg, var(--bgTop) 0%, var(--bgMid) 45%, var(--bgBot) 100%);
}

header[data-testid="stHeader"]{ background: transparent; }
.block-container{ padding-top: 22px; }

.main .block-container{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 22px;
  padding: 18px 18px 22px 18px;
  box-shadow: 0 14px 36px rgba(0,0,0,0.18);
  backdrop-filter: blur(10px);
}

h1, h2, h3 { color: var(--text); font-weight: 900; }
.stCaption, p { color: var(--muted); }

section[data-testid="stSidebar"]{
  background: rgba(11,46,42,0.45) !important;
  border-left: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(10px);
}
section[data-testid="stSidebar"] *{ color: var(--text) !important; }
section[data-testid="stSidebar"] label{
  color: rgba(234,247,238,0.88) !important;
  font-weight: 800 !important;
}

.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
}

.kpi{
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 18px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
}
.kpi .title{
  font-size: 13px;
  color: rgba(234,247,238,0.82);
  font-weight: 900;
  letter-spacing: .2px;
}
.kpi .value{
  margin-top: 8px;
  font-size: 34px;
  color: var(--text);
  font-weight: 950;
  line-height: 1;
}
.kpi .sub{
  margin-top: 8px;
  font-size: 12px;
  color: rgba(234,247,238,0.72);
  font-weight: 700;
}
.kpi .delta{
  margin-top: 10px;
  font-size: 12px;
  font-weight: 800;
  color: rgba(234,247,238,0.80);
}

.badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 900;
  border: 1px solid rgba(255,255,255,0.22);
  background: rgba(255,255,255,0.10);
  color: var(--text);
}
.badge-low{ background: rgba(221,239,225,0.18); border-color: rgba(221,239,225,0.35); }
.badge-med{ background: rgba(255,236,179,0.18); border-color: rgba(255,236,179,0.35); }
.badge-high{ background: rgba(255,199,199,0.18); border-color: rgba(255,199,199,0.35); }

.stTabs [data-baseweb="tab-list"]{ gap: 10px; }
.stTabs [data-baseweb="tab"]{
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 16px;
  padding: 10px 14px;
  color: rgba(234,247,238,0.80);
  font-weight: 900;
}
.stTabs [aria-selected="true"]{
  background: rgba(221,239,225,0.22);
  border-color: rgba(221,239,225,0.38);
  color: var(--text);
  box-shadow: 0 10px 24px rgba(0,0,0,0.18);
}

[data-testid="stDataFrame"]{
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
}

.js-plotly-plot, .plot-container{ background: transparent !important; }

.stButton>button{
  background: rgba(221,239,225,0.85) !important;
  color: #0B2E2A !important;
  border: 1px solid rgba(221,239,225,0.65) !important;
  border-radius: 14px !important;
  padding: 10px 16px !important;
  font-weight: 950 !important;
}
.stButton>button:hover{ background: rgba(221,239,225,0.98) !important; }

div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.10) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
  border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

PLOT_SEQ = ["#DDEFE1", "#CFE7D7", "#9CCFB9", "#2A6B5E"]

# =========================
# أسماء الشهور بالعربي
# =========================
AR_MONTHS = {
    1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل",
    5: "مايو", 6: "يونيو", 7: "يوليو", 8: "أغسطس",
    9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر"
}
MONTH_ORDER = list(AR_MONTHS.values())
MONTH_NAME_TO_NUM = {v: k for k, v in AR_MONTHS.items()}

# =========================
# دوال مساعدة
# =========================
def style_fig(fig):
    fig.update_layout(
        plot_bgcolor="rgba(255,255,255,0.06)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        font=dict(color="#EAF7EE", family="IBM Plex Sans Arabic"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="rgba(234,247,238,0.85)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.10)", zeroline=False, color="rgba(234,247,238,0.85)")
    return fig

def format_delta(curr, prev, kind="percent"):
    if prev is None:
        return "لا تتوفر بيانات مقارنة"
    if kind == "percent":
        diff = (curr - prev) * 100
        sign = "تحسن" if diff < 0 else "تدهور" if diff > 0 else "ثبات"
        return f"مقارنة باليوم السابق: {abs(diff):.1f}% ({sign})"
    if kind == "number":
        diff = curr - prev
        sign = "ارتفاع" if diff > 0 else "انخفاض" if diff < 0 else "ثبات"
        return f"مقارنة باليوم السابق: {abs(diff):,.0f} ({sign})"
    if kind == "ms":
        diff = curr - prev
        sign = "تحسن" if diff < 0 else "تدهور" if diff > 0 else "ثبات"
        return f"مقارنة باليوم السابق: {abs(diff):.0f} ms ({sign})"
    return "—"

def style_table(df: pd.DataFrame):
    return (
        df.style
        .set_properties(**{"text-align": "right", "font-family": "IBM Plex Sans Arabic"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align","right"), ("font-weight","800")]},
            {"selector": "td", "props": [("text-align","right")]},
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "rgba(255,255,255,0.05)")]},
        ])
    )

# =========================
# العنوان
# =========================
st.title("منصة كشف")
st.caption("لوحة مؤشرات التعثر وجودة رحلة الخدمات الرقمية (بيانات مجهولة الهوية - نموذج أولي)")

# =========================
# تحميل البيانات
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/events.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

# تجهيز سنة/شهر للفلترة
df["year"] = df["timestamp"].dt.year
df["month_num"] = df["timestamp"].dt.month
df["month_name"] = df["month_num"].map(AR_MONTHS)

available_years = sorted(df["year"].unique().tolist())
available_months = [m for m in MONTH_ORDER if m in df["month_name"].unique().tolist()]

# =========================
# Sidebar Filters (مع فلتر شهر مثل الصورة)
# =========================
with st.sidebar:
    st.markdown("### المرشحات")
    entity = st.selectbox("الجهة", ["الكل"] + sorted(df["entity"].unique().tolist()))
    service = st.selectbox("الخدمة", ["الكل"] + sorted(df["service"].unique().tolist()))
    device = st.selectbox("الجهاز", ["الكل"] + sorted(df["device"].unique().tolist()))

    st.markdown("---")
    year_sel = st.selectbox("السنة", ["الكل"] + [str(y) for y in available_years])
    month_sel = st.selectbox("الشهر", ["الكل"] + available_months)

    st.markdown("---")
    threshold = st.slider("حد اعتبار المشكلة (حالات متشابهة)", 10, 200, 50, 10)
    st.caption("لا يتم اعتبار المشكلة مؤشرًا إلا عند تكرارها وفق حد أدنى محدد.")

# =========================
# تطبيق الفلاتر (تشمل الشهر/السنة)
# =========================
f = df.copy()

if entity != "الكل":
    f = f[f["entity"] == entity]
if service != "الكل":
    f = f[f["service"] == service]
if device != "الكل":
    f = f[f["device"] == device]

if year_sel != "الكل":
    f = f[f["year"] == int(year_sel)]
if month_sel != "الكل":
    f = f[f["month_num"] == MONTH_NAME_TO_NUM[month_sel]]

# =========================
# بناء Sessions Features
# =========================
g = f.groupby(["session_id", "entity", "service", "device"], as_index=False)
sessions = g.agg(
    start_time=("timestamp", "min"),
    end_time=("timestamp", "max"),
    events=("event_type", "count"),
    errors=("event_type", lambda x: (x == "خطأ").sum()),
    retries=("event_type", lambda x: (x == "إعادة_محاولة").sum()),
    drop=("event_type", lambda x: (x == "انسحاب").any()),
    success=("event_type", lambda x: (x == "نجاح").any()),
    avg_latency=("latency_ms", "mean"),
    max_latency=("latency_ms", "max"),
    avg_step_time=("duration_sec", "mean"),
)
if len(sessions) > 0:
    sessions["total_time_sec"] = (sessions["end_time"] - sessions["start_time"]).dt.total_seconds()
else:
    sessions["total_time_sec"] = 0

sessions["drop"] = sessions["drop"].astype(int) if len(sessions) else 0
sessions["success"] = sessions["success"].astype(int) if len(sessions) else 0

# =========================
# KPIs
# =========================
total_sessions = int(sessions["session_id"].nunique()) if len(sessions) else 0
drop_rate = float(sessions["drop"].mean()) if total_sessions else 0.0
error_rate = float(f["event_type"].eq("خطأ").sum() / max(total_sessions, 1)) if len(f) else 0.0

avg_latency = f.loc[f["event_type"].eq("نداء_API"), "latency_ms"].mean() if len(f) else 0.0
avg_latency = 0.0 if pd.isna(avg_latency) else float(avg_latency)

severity = min(100.0, (error_rate*100*1.4) + (drop_rate*100*1.2) + (min(avg_latency, 1500)/1500*40))

def severity_tag(x):
    if x >= 70:
        return '<span class="badge badge-high">عالي</span>'
    if x >= 40:
        return '<span class="badge badge-med">متوسط</span>'
    return '<span class="badge badge-low">منخفض</span>'

# =========================
# KPI Cards
# =========================
k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1])

k1.markdown(f"""
<div class="kpi">
  <div class="title">شدة المشكلة</div>
  <div class="value">{severity:.0f}/100</div>
  <div class="sub">{severity_tag(severity)}</div>
  <div class="delta">مؤشر مركّب: أخطاء + انسحاب + بطء</div>
</div>
""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="kpi">
  <div class="title">متوسط زمن الاستجابة</div>
  <div class="value">{avg_latency:.0f} ms</div>
  <div class="sub">متوسط زمن نداءات API</div>
  <div class="delta">ضمن المرشحات المحددة</div>
</div>
""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="kpi">
  <div class="title">معدل الأخطاء</div>
  <div class="value">{error_rate*100:.1f}%</div>
  <div class="sub">تقريبي بالنسبة للجلسات</div>
  <div class="delta">ضمن المرشحات المحددة</div>
</div>
""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="kpi">
  <div class="title">معدل الانسحاب</div>
  <div class="value">{drop_rate*100:.1f}%</div>
  <div class="sub">نسبة الجلسات غير المكتملة</div>
  <div class="delta">ضمن المرشحات المحددة</div>
</div>
""", unsafe_allow_html=True)

k5.markdown(f"""
<div class="kpi">
  <div class="title">إجمالي الجلسات</div>
  <div class="value">{total_sessions:,}</div>
  <div class="sub">ضمن المرشحات المختارة</div>
  <div class="delta">قياس مجمّع</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["نظرة عامة", "نقاط الانسحاب", "كشف الشذوذ", "أنماط التعثر", "الإنذارات"]
)

# =========================
# Tab 1: Overview (شهور بأسماء عربي)
# =========================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("اتجاه المؤشرات عبر الشهور")

    d = f.copy()
    if len(d) == 0:
        st.info("لا توجد بيانات ضمن المرشحات الحالية.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        d["سنة"] = d["timestamp"].dt.year
        d["شهر_رقم"] = d["timestamp"].dt.month
        d["شهر"] = d["شهر_رقم"].map(AR_MONTHS)
        d["شهر_بداية"] = d["timestamp"].dt.to_period("M").dt.to_timestamp()

        monthly = d.groupby(["شهر_بداية", "سنة", "شهر"], as_index=False).agg(
            جلسات=("session_id", "nunique"),
            أخطاء=("event_type", lambda x: (x == "خطأ").sum()),
            انسحاب=("event_type", lambda x: (x == "انسحاب").sum()),
        )
        monthly["معدل_الأخطاء"] = monthly["أخطاء"] / monthly["جلسات"].clip(lower=1)
        monthly["معدل_الانسحاب"] = monthly["انسحاب"] / monthly["جلسات"].clip(lower=1)

        monthly["الفترة"] = monthly["شهر"] + " " + monthly["سنة"].astype(str)
        order = monthly.sort_values("شهر_بداية")["الفترة"].tolist()
        monthly["الفترة"] = pd.Categorical(monthly["الفترة"], categories=order, ordered=True)

        fig = px.line(
            monthly.sort_values("شهر_بداية"),
            x="الفترة",
            y=["معدل_الأخطاء", "معدل_الانسحاب"],
            markers=True,
            color_discrete_sequence=PLOT_SEQ
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # توصيات
    st.markdown('<div class="card" style="margin-top:10px;">', unsafe_allow_html=True)
    st.subheader("توصيات آلية")
    recs = []
    upload_times = f[(f["step"] == "رفع_مستند") & (f["event_type"] == "زمن_خطوة")]["duration_sec"] if len(f) else []

    if drop_rate > 0.18:
        recs.append("تبسيط النموذج وإضافة حفظ تلقائي للخطوات لتقليل الانسحاب.")
    if avg_latency > 600:
        recs.append("تحسين زمن الاستجابة عبر مراجعة نقاط التكامل وتفعيل التخزين المؤقت (Caching).")
    if len(upload_times) > 0 and np.median(upload_times) > 25:
        recs.append("تحسين خطوة رفع المستندات عبر مؤشر تقدم وتوضيح المتطلبات وتقليل حجم الملفات.")
    if error_rate > 0.08:
        recs.append("تحسين رسائل الخطأ والتحقق من المدخلات قبل الإرسال لتقليل الفشل المتكرر.")
    if severity >= 70:
        recs.append("تفعيل إجراءات الاستجابة للحوادث ومراقبة لحظية عند ارتفاع الشدة.")
    if not recs:
        recs.append("المؤشرات ضمن النطاق الطبيعي حالياً. يوصى بالمتابعة الدورية.")

    for r in recs:
        st.write(f"- {r}")
    st.caption("التوصيات في النموذج الأولي مبنية على قواعد قابلة للتطوير.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Tab 2: Funnel
# =========================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("تحليل نقاط الانسحاب (Funnel)")

    steps = ["بدء", "تعبئة_نموذج", "رفع_مستند", "مراجعة", "إرسال"]
    step_views = f[f["event_type"] == "عرض_خطوة"] if len(f) else f
    counts = step_views.groupby("step")["session_id"].nunique().reindex(steps).fillna(0).astype(int) if len(step_views) else pd.Series([0]*len(steps), index=steps)

    funnel = pd.DataFrame({"الخطوة": steps, "عدد_الجلسات": counts.values})
    funnel["نسبة_التحويل"] = (funnel["عدد_الجلسات"] / max(funnel["عدد_الجلسات"].max(), 1) * 100).round(1)

    st.dataframe(style_table(funnel), use_container_width=True)

    fig2 = px.bar(funnel, x="الخطوة", y="عدد_الجلسات", color_discrete_sequence=[PLOT_SEQ[1]])
    st.plotly_chart(style_fig(fig2), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Tab 3: Anomaly Detection
# =========================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("كشف الشذوذ (ارتفاع مفاجئ في الأخطاء/البطء)")

    if len(f) == 0:
        st.info("لا توجد بيانات ضمن المرشحات الحالية.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        tmp = f.copy()
        tmp["ساعة"] = tmp["timestamp"].dt.floor("H")

        agg = tmp.groupby(["ساعة", "entity", "service"], as_index=False).agg(
            جلسات=("session_id", "nunique"),
            أخطاء=("event_type", lambda x: (x == "خطأ").sum()),
            متوسط_زمن_API=("latency_ms", "mean"),
            انسحاب=("event_type", lambda x: (x == "انسحاب").sum())
        )
        agg["معدل_الأخطاء"] = agg["أخطاء"] / agg["جلسات"].clip(lower=1)
        agg["معدل_الانسحاب"] = agg["انسحاب"] / agg["جلسات"].clip(lower=1)
        agg = agg.fillna(0)

        feats = agg[["معدل_الأخطاء", "متوسط_زمن_API", "معدل_الانسحاب"]]
        iso = IsolationForest(random_state=7, contamination=0.03)
        preds = iso.fit_predict(feats)
        agg["شذوذ"] = np.where(preds == -1, "نعم", "لا")

        st.dataframe(style_table(agg.sort_values("شذوذ", ascending=False).head(40)), use_container_width=True)

        an_yes = agg[agg["شذوذ"] == "نعم"]
        if len(an_yes) > 0:
            fig3 = px.scatter(
                an_yes, x="ساعة", y="معدل_الأخطاء", size="جلسات",
                hover_data=["entity", "service", "متوسط_زمن_API", "معدل_الانسحاب"],
                color_discrete_sequence=[PLOT_SEQ[0]]
            )
            st.plotly_chart(style_fig(fig3), use_container_width=True)
        else:
            st.info("لا توجد حالات شذوذ واضحة ضمن بيانات النموذج الأولي.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Tab 4: Clustering
# =========================
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("أنماط التعثر (Clustering)")

    if len(sessions) < 5:
        st.info("البيانات غير كافية لتجميع موثوق. جرّبي توسيع الفترة أو زيادة البيانات.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        X = sessions[["errors", "retries", "total_time_sec", "max_latency"]].fillna(0)
        km = KMeans(n_clusters=4, random_state=7, n_init="auto")
        sessions["المجموعة"] = km.fit_predict(X)

        summary = sessions.groupby("المجموعة", as_index=False).agg(
            عدد_الجلسات=("session_id", "nunique"),
            متوسط_الوقت=("total_time_sec", "mean"),
            متوسط_الأخطاء=("errors", "mean"),
            متوسط_المحاولات=("retries", "mean"),
            معدل_الانسحاب=("drop", "mean"),
        )

        st.dataframe(style_table(summary), use_container_width=True)

        fig4 = px.bar(summary, x="المجموعة", y="عدد_الجلسات", color_discrete_sequence=[PLOT_SEQ[1]])
        st.plotly_chart(style_fig(fig4), use_container_width=True)

        st.caption("كل مجموعة تمثل نمطًا متشابهًا من التعثر عبر الجلسات.")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Tab 5: Alerts
# =========================
with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f"الإنذارات عند تكرار المشكلة (≥ {threshold} حالة متشابهة)")

    if len(f) == 0:
        st.info("لا توجد بيانات ضمن المرشحات الحالية.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        issues = f[f["event_type"].isin(["خطأ", "انسحاب"])].copy()
        issues["error_code"] = issues["error_code"].fillna("NA")

        issue_counts = issues.groupby(["entity", "service", "step", "event_type", "error_code"], as_index=False).agg(
            الحالات=("session_id", "nunique"),
            أول_ظهور=("timestamp", "min"),
            آخر_ظهور=("timestamp", "max"),
        )

        alerts = issue_counts[issue_counts["الحالات"] >= threshold].sort_values("الحالات", ascending=False)

        if len(alerts) == 0:
            st.info("لا توجد إنذارات حسب الحد الحالي. يمكن خفض الحد من الشريط الجانبي.")
        else:
            st.dataframe(style_table(alerts), use_container_width=True)

            top10 = alerts.head(10).copy()
            top10["وصف"] = top10["service"] + " | " + top10["step"] + " | " + top10["event_type"] + " | " + top10["error_code"]
            fig5 = px.bar(top10, x="الحالات", y="وصف", orientation="h", color_discrete_sequence=[PLOT_SEQ[0]])
            st.plotly_chart(style_fig(fig5), use_container_width=True)

        st.caption("تم بناء الإنذار بناءً على تكرار المشكلة لضمان دلالة إحصائية قبل التصعيد.")
        st.markdown("</div>", unsafe_allow_html=True)